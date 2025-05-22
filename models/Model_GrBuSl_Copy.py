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
    def __init__(self, input_emb_dim, sup_hidden_layers = [64, 32, 1]):
        super(SupervisedVisionNetwork, self).__init__()
        self.tower1 = SimpleDenseNetwork(input_emb_dim, sup_hidden_layers, top_no_act = True)
        
    def forward(self, x, user_type_onehot, idx_vec, reward_start, reward_range):
        tower1_out = self.tower1(x)
        return tower1_out

class CriticNetwork(nn.Module):
    def __init__(self, input_emb_dim, critic_a_hidden_layers = [64, 32, 2]):
        super(CriticNetwork, self).__init__()
        self.ln = nn.LayerNorm(input_emb_dim)
        self.tower1 = SimpleDenseNetwork(input_emb_dim, critic_a_hidden_layers, top_no_act = True)
        
    def forward(self, base_tower, labels, user_type_onehot):
        base_tower = self.ln(base_tower)
        tower1_out = self.tower1(base_tower)
        pred_q_value = tower1_out
        return pred_q_value

class TimeCriticNetwork(nn.Module):
    def __init__(self, base_output_dim):
        super(TimeCriticNetwork, self).__init__()
        # 这俩作用是取min
        self.critic1 = CriticNetwork(base_output_dim)
        self.critic2 = CriticNetwork(base_output_dim)
        
    def _get_cur_actions(self, reco_ban):
        actions = []
        for i in reco_ban:
            if i == 0:
                actions.append([1.0, 0.0])
            else:
                actions.append([0.0, 1.0])
        return torch.Tensor(actions).cuda(device_controller.get_device())

    def forward(self, cur_x, nxt_x, cur_labels, nxt_labels, not_final, user_type_onehot, target_network):
        # define Hyper params
        alpha = 1.0
        delta = 0.1
        cur_q1_value = self.critic1(cur_x, cur_labels, user_type_onehot)
        cur_q2_value = self.critic2(cur_x, cur_labels, user_type_onehot)
        
        cur_q_value = torch.min(cur_q1_value, cur_q2_value)
        cur_action = self._get_cur_actions(cur_labels['reco_ban'])
        cur_q1_action_value = torch.sum(cur_q1_value * cur_action, dim=1, keepdim=True)
        cur_q2_action_value = torch.sum(cur_q2_value * cur_action, dim=1, keepdim=True)
        nxt_q1_value = target_network.critic1(nxt_x, nxt_labels, user_type_onehot)
        nxt_q2_value = target_network.critic2(nxt_x, nxt_labels, user_type_onehot)
        nxt_q_value = torch.min(nxt_q1_value, nxt_q2_value)
        nxt_q_value, _ = torch.max(nxt_q_value, dim =1 , keepdim=True)
        cur_time_delta = torch.Tensor(cur_labels['time_delta']).reshape((-1, 1)).cuda(device_controller.get_device())
        cur_time_delta_reward = F.sigmoid(0.1 * cur_time_delta)
        q_label = cur_time_delta_reward + 0.9 * torch.Tensor(not_final).view(-1, 1).cuda(device_controller.get_device()) * nxt_q_value.detach()
        q1_loss = new_huber_loss(q_label, cur_q1_action_value, 1.0, alpha, delta)
        q2_loss = new_huber_loss(q_label, cur_q2_action_value, 1.0, alpha, delta)
        rl_loss = torch.sum(q1_loss) + torch.sum(q2_loss)
        return cur_q_value, rl_loss

class LiveRecoModelGrBuSlCopy(nn.Module):
    def __init__(self, 
                 input_network = None,
                 base_output_dim = 128
                 ):
        super(LiveRecoModelGrBuSlCopy, self).__init__()
        self.input_network = input_network.cuda(device_controller.get_device())
        self.actor_network = ActorNetwork(base_output_dim).cuda(device_controller.get_device())
        self.time_critic1 = TimeCriticNetwork(base_output_dim).cuda(device_controller.get_device())
        self.time_critic2 = TimeCriticNetwork(base_output_dim).cuda(device_controller.get_device())
    def _get_cur_actions(self, reco_ban):
        actions = []
        for i in reco_ban:
            if i == 0:
                actions.append([1.0, 0.0])
            else:
                actions.append([0.0, 1.0])
        return torch.Tensor(actions).cuda(device_controller.get_device())

    @torch.no_grad()
    def _copy_critic_parameters(self):
        self.time_critic2.load_state_dict(self.time_critic1.state_dict())
        self.time_critic2.eval()
    def forward(self, input):
        cur_features, nxt_features, cur_labels, nxt_labels, not_final  = input['cur_features'],input['nxt_features'],input['cur_labels'],input['nxt_labels'], input['not_final']
        cur_common_input_res = self.input_network(cur_features)

        user_type = cur_features['user_type'] # list: len = bs
        user_type_indices = torch.tensor(user_type, dtype=torch.long).cuda(device_controller.get_device()) - 1       # [1, 6]--->范围是0~5
        user_type_onehot = F.one_hot(user_type_indices, num_classes=6)  # [batch_size, user_type_num]
        cur_action_logits = self.actor_network(cur_common_input_res.detach(), user_type_onehot)
        cur_action_probs = F.softmax(cur_action_logits, dim=1)
        nxt_common_input_res = self.input_network(nxt_features)
        
        #交替训练 ----> 改成复制。
        time_q_value1, time_rl_loss1 = self.time_critic1(cur_common_input_res, nxt_common_input_res, cur_labels, nxt_labels, not_final, user_type_onehot, self.time_critic2)

        # 计算总的 q 值、动作概率和损失
        cur_action = self._get_cur_actions(cur_labels['reco_ban'])
        cur_time_reward = torch.Tensor(cur_labels['time_reward']).cuda(device_controller.get_device())
        time_actor_weight = 2.0
        tot_q_value = time_actor_weight * time_q_value1
        tot_q_probs = torch.softmax(tot_q_value, dim=1)

        # 计算损失
        ce_loss = torch.sum(F.cross_entropy(cur_action_logits, tot_q_probs, reduction='none').view(-1, 1))
        actor_loss = ce_loss
        rl_loss = time_rl_loss1
        critic_loss = rl_loss
        sup_loss = torch.zeros_like(rl_loss).cuda(device_controller.get_device())

        return cur_action, cur_time_reward, cur_action_probs, actor_loss, rl_loss, critic_loss, sup_loss

    def get_probs(self, input):
        cur_features, nxt_features, cur_labels, nxt_labels, not_final  = input['cur_features'],input['nxt_features'],input['cur_labels'],input['nxt_labels'], input['not_final']
        cur_common_input_res = self.input_network(cur_features)

        user_type = cur_features['user_type'] # list: len = bs
        user_type_indices = torch.tensor(user_type, dtype=torch.long).cuda(device_controller.get_device()) - 1       # [1, 6]--->范围是0~5
        user_type_onehot = F.one_hot(user_type_indices, num_classes=6)  # [batch_size, user_type_num]
        cur_action_logits = self.actor_network(cur_common_input_res.detach(), user_type_onehot)
        cur_action_probs = F.softmax(cur_action_logits, dim=1)
        return cur_action_probs
