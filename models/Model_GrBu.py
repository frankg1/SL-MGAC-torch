import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch import nn
import torch
import torch.nn.functional as F
from random import randint
import numpy as np
from .Base import SimpleDenseNetwork
from .Base import device_controller
class ActorNetwork(nn.Module):
    def __init__(self, input_emb_dim, actor_user_hidden_layers = [128, 63, 31, 2]):
        super(ActorNetwork, self).__init__()
        self.tower1 = SimpleDenseNetwork(input_emb_dim, actor_user_hidden_layers, top_no_act = True)
        self.ln = nn.LayerNorm(input_emb_dim)
        
    def forward(self, x, user_type_onehot):
        x = self.ln(x) 
        return self.tower1(x)

def new_huber_loss(label, pred, weight, alpha_val, delta_val):
    residual = torch.abs(label - pred)
    min_residual = torch.minimum(residual, torch.tensor(delta_val).cuda(device_controller.get_device()))
    huber_loss = weight * (0.5 * torch.square(min_residual) + alpha_val * (residual - min_residual))
    return huber_loss

class StepCounter:
    def __init__(self):
        self.step = 0
    def get_step(self):
        return self.step
    def increment_step(self):
        self.step += 1
step_counter = StepCounter()

class SupervisedVisionNetwork(nn.Module):
    def __init__(self, input_emb_dim, sup_hidden_layers = [64, 32, 2]):
        super(SupervisedVisionNetwork, self).__init__()
        self.tower1 = SimpleDenseNetwork(input_emb_dim, sup_hidden_layers, top_no_act = True)
        
    def forward(self, x, user_type_onehot, idx_vec, reward_start, reward_range):
        tower1_out = self.tower1(x)
        return tower1_out

class CriticNetwork(nn.Module):
    def __init__(self, input_emb_dim, critic_a_hidden_layers = [64, 32, 2]):
        super(CriticNetwork, self).__init__()
        self.ln = nn.LayerNorm(input_emb_dim)
        self.live_sup_tower = SupervisedVisionNetwork(input_emb_dim, sup_hidden_layers = [64, 32, 2])
        self.photo_sup_tower = SupervisedVisionNetwork(input_emb_dim, sup_hidden_layers = [64, 32, 2])
        self.tower1 = SimpleDenseNetwork(input_emb_dim, critic_a_hidden_layers, top_no_act = True)
        
    def forward(self, base_tower, labels, user_type_onehot):
        # 拿到Labels
        time_idx = labels["time_idx_vec"]
        photo_idx = labels["photo_idx_vec"]
        live_start = labels["time_reward_start"]
        live_range = labels["time_reward_range"]
        photo_start = labels["photo_reward_start"]
        photo_range = labels["photo_reward_range"]

        base_tower = self.ln(base_tower)
        #下面是过监督层。
        pred_live_time = self.live_sup_tower(base_tower, user_type_onehot, time_idx, live_start, live_range)
        pred_photo_time = self.photo_sup_tower(base_tower, user_type_onehot, photo_idx, photo_start, photo_range)
        pred_time_delta = pred_live_time - pred_photo_time

        pred_reward = torch.sigmoid(0.1 * pred_time_delta) 
    
        tower1_out = self.tower1(base_tower)
        a_user_tower = tower1_out

        pred_q_value = pred_reward.detach() + 0.9 * F.elu(a_user_tower)
        return pred_reward, pred_time_delta, pred_live_time, pred_photo_time, pred_q_value

class TimeCriticNetwork(nn.Module):
    def __init__(self, base_output_dim):
        super(TimeCriticNetwork, self).__init__()
        # 这俩作用是取min
        self.critic1 = CriticNetwork(base_output_dim)
        self.critic2 = CriticNetwork(base_output_dim)
        
    def set_target_network(self, target_network):
        self.target_network = target_network
        
    def _get_cur_actions(self, reco_ban):
        actions = []
        for i in reco_ban:
            if i == 0:
                actions.append([1.0, 0.0])
            else:
                actions.append([0.0, 1.0])
        return torch.Tensor(actions).cuda(device_controller.get_device())

    def forward(self, cur_x, nxt_x, cur_labels, nxt_labels, not_final, user_type_onehot):
        # define Hyper params
        alpha_time = 1.0
        delta_time = 0.1
        alpha_photo = 2.0
        delta_photo = 0.1
        alpha = 1.0
        delta = 0.1
        
        #pred_reward, pred_time_delta, pred_live_time, pred_photo_time, pred_q_value
        cur_q1_pred_reward, cur_q1_pred_time_delta, cur_q1_pred_live_time, cur_q1_pred_photo_time, cur_q1_value = self.critic1(cur_x, cur_labels, user_type_onehot)
        cur_q2_pred_reward, cur_q2_pred_time_delta, cur_q2_pred_live_time, cur_q2_pred_photo_time, cur_q2_value = self.critic2(cur_x, cur_labels, user_type_onehot)
        
        cur_q_value = torch.min(cur_q1_value, cur_q2_value)
        cur_q_argmax = torch.argmax(cur_q_value, dim = 1)
        
        #先得到cur_action，如果(cur_labels)reco_ban ==1 cur_action={0.0 1.0} 否则=={1.0, 0.0}
        cur_action = self._get_cur_actions(cur_labels['reco_ban'])
        
        cur_q1_action_pred_reward = torch.sum(cur_q1_pred_reward * cur_action, dim=1, keepdim=True)
        cur_q2_action_pred_reward = torch.sum(cur_q2_pred_reward * cur_action, dim=1, keepdim=True)
        
        cur_q1_action_value = torch.sum(cur_q1_value * cur_action, dim=1, keepdim=True)
        cur_q2_action_value = torch.sum(cur_q2_value * cur_action, dim=1, keepdim=True)        
        
        nxt_q1_pred_reward, nxt_q1_pred_time_delta, nxt_q1_pred_live_time, nxt_q1_pred_photo_time, nxt_q1_value = self.target_network.critic1(nxt_x, nxt_labels, user_type_onehot)
        
        nxt_q2_pred_reward, nxt_q2_pred_time_delta, nxt_q2_pred_live_time, nxt_q2_pred_photo_time, nxt_q2_value = self.target_network.critic2(nxt_x, nxt_labels, user_type_onehot)
        
        nxt_q_value = torch.min(nxt_q1_value, nxt_q2_value)
        nxt_q_value, _ = torch.max(nxt_q_value, dim =1 , keepdim=True)
        cur_time_delta = torch.Tensor(cur_labels['time_delta']).reshape((-1, 1)).cuda(device_controller.get_device())
        cur_time_delta_reward = F.sigmoid(0.1 * cur_time_delta)
        q_label = cur_time_delta_reward + 0.9 * torch.Tensor(not_final).view(-1, 1).cuda(device_controller.get_device()) * nxt_q_value.detach()
        
        q1_reward_loss = new_huber_loss(cur_time_delta_reward, cur_q1_action_pred_reward, 1 * 1.0, alpha_time, delta_time)
        q2_reward_loss = new_huber_loss(cur_time_delta_reward, cur_q2_action_pred_reward, 1 * 1.0, alpha_time, delta_time)
        q1_loss = new_huber_loss(q_label, cur_q1_action_value, 1.0, alpha, delta)
        q2_loss = new_huber_loss(q_label, cur_q2_action_value, 1.0, alpha, delta)

        time_delta_sup_loss = torch.sum(q1_reward_loss) + torch.sum(q2_reward_loss)
        sup_loss =  time_delta_sup_loss
        rl_loss = torch.sum(q1_loss) + torch.sum(q2_loss)

        return cur_q_value, time_delta_sup_loss, sup_loss, rl_loss

class LiveRecoModelGrBu(nn.Module):
    def __init__(self, 
                 input_network = None,
                 base_output_dim = 128
                 ):
        super(LiveRecoModelGrBu, self).__init__()
        self.input_network = input_network.cuda(device_controller.get_device())
        self.actor_network = ActorNetwork(base_output_dim).cuda(device_controller.get_device())
        self.time_critic1 = TimeCriticNetwork(base_output_dim).cuda(device_controller.get_device())
        self.time_critic2 = TimeCriticNetwork(base_output_dim).cuda(device_controller.get_device())
        self.time_critic1.set_target_network(self.time_critic2)
        self.time_critic2.set_target_network(self.time_critic1)
    def _get_cur_actions(self, reco_ban):
        actions = []
        for i in reco_ban:
            if i == 0:
                actions.append([1.0, 0.0])
            else:
                actions.append([0.0, 1.0])
        return torch.Tensor(actions).cuda(device_controller.get_device())
    
    def forward(self, input):
        cur_features, nxt_features, cur_labels, nxt_labels, not_final  = input['cur_features'],input['nxt_features'],input['cur_labels'],input['nxt_labels'], input['not_final']
        # print(not_final)
        cur_common_input_res = self.input_network(cur_features)
        
        user_type = cur_features['user_type'] # list: len = bs
        user_type_indices = torch.tensor(user_type, dtype=torch.long).cuda(device_controller.get_device()) - 1       # [1, 6]--->范围是0~5
        user_type_onehot = F.one_hot(user_type_indices, num_classes=6)  # [batch_size, user_type_num]
        cur_action_logits = self.actor_network(cur_common_input_res.detach(), user_type_onehot)
        cur_action_probs = F.softmax(cur_action_logits, dim=1)
        
        nxt_common_input_res = self.input_network(nxt_features)
        
        #交替训练 ----> 改成复制。
        # cur_q_value, time_delta_sup_loss, sup_loss, rl_loss
        time_q_value1, time_delta_sup_loss1, sup_loss1, time_rl_loss1 = self.time_critic1(cur_common_input_res, nxt_common_input_res, cur_labels, nxt_labels, not_final, user_type_onehot)
        
        time_q_value2, time_delta_sup_loss2, sup_loss2, time_rl_loss2 = self.time_critic2(cur_common_input_res, nxt_common_input_res, cur_labels, nxt_labels, not_final, user_type_onehot)
        
        # 外面定义step，实现交替训练的逻辑
        # 再把源代码复制过来
        cur_action = self._get_cur_actions(cur_labels['reco_ban'])
        cur_time_reward =torch.Tensor(cur_labels['time_reward']).cuda(device_controller.get_device()) 
        
        step_mod = step_counter.get_step() % 2
        
        time_q_value = (1.0 - step_mod) * time_q_value1 + step_mod * time_q_value2
        time_delta_sup_loss = (1.0 - step_mod) * time_delta_sup_loss1 + step_mod * time_delta_sup_loss2
        sup_loss = (1.0 - step_mod) * sup_loss1 + step_mod * sup_loss2
        time_rl_loss = (1.0 - step_mod) * time_rl_loss1 + step_mod * time_rl_loss2
        # print('time_rl_loss: ', time_rl_loss1.item(), time_rl_loss2.item())
        
        cur_action_prob = torch.sum(cur_action_probs * cur_action, dim=1, keepdim=True)
        cur_log_action_prob = torch.log(cur_action_prob + 1e-10)
        
        time_actor_weight = 2.0
        ctr_loss_weight = 100.0
        cur_reg_weight = 0.0
        
        tot_q_value = time_actor_weight * time_q_value
        tot_q_probs = torch.softmax(tot_q_value, dim=1)
        tot_q_action_prob = torch.sum(tot_q_probs * cur_action, dim=1, keepdim=True)
        ones = torch.ones_like(tot_q_action_prob)
        ce_loss = torch.sum(F.cross_entropy(cur_action_logits, tot_q_probs, reduction='none').view(-1, 1))
        actor_loss = ce_loss
        rl_loss = time_rl_loss
        critic_loss = sup_loss + rl_loss
        
        return cur_action, cur_time_reward, cur_action_probs, actor_loss, time_rl_loss, critic_loss, sup_loss
    def get_probs(self, input):
        cur_features, nxt_features, cur_labels, nxt_labels, not_final  = input['cur_features'],input['nxt_features'],input['cur_labels'],input['nxt_labels'], input['not_final']
        # print(not_final)
        cur_common_input_res = self.input_network(cur_features)

        user_type = cur_features['user_type'] # list: len = bs
        user_type_indices = torch.tensor(user_type, dtype=torch.long).cuda(device_controller.get_device()) - 1       # [1, 6]--->范围是0~5
        user_type_onehot = F.one_hot(user_type_indices, num_classes=6)  # [batch_size, user_type_num]
        cur_action_logits = self.actor_network(cur_common_input_res.detach(), user_type_onehot)
        cur_action_probs = F.softmax(cur_action_logits, dim=1)
        return cur_action_probs
