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
        self.tower2 = SimpleDenseNetwork(input_emb_dim, actor_user_hidden_layers, top_no_act = True)
        self.tower3 = SimpleDenseNetwork(input_emb_dim, actor_user_hidden_layers, top_no_act = True)
        self.tower4 = SimpleDenseNetwork(input_emb_dim, actor_user_hidden_layers, top_no_act = True)
        self.tower5 = SimpleDenseNetwork(input_emb_dim, actor_user_hidden_layers, top_no_act = True)
        self.tower6 = SimpleDenseNetwork(input_emb_dim, actor_user_hidden_layers, top_no_act = True)
        self.ln = nn.LayerNorm(input_emb_dim)
        
    def forward(self, x, user_type_onehot):
        x = self.ln(x) 
        tower1_out = self.tower1(x).unsqueeze(2)
        tower2_out = self.tower2(x).unsqueeze(2)
        tower3_out = self.tower3(x).unsqueeze(2)
        tower4_out = self.tower4(x).unsqueeze(2)
        tower5_out = self.tower5(x).unsqueeze(2)
        tower6_out = self.tower6(x).unsqueeze(2)
        user_flag = user_type_onehot.unsqueeze(1)
        user_tower_list = torch.concat([tower1_out,tower2_out,tower3_out,tower4_out,tower5_out,tower6_out], dim =2) #[bs,dim, 6]
        return torch.mul(user_flag, user_tower_list).sum(dim = 2)

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
    def __init__(self, input_emb_dim, sup_hidden_layers = [64, 32, 8]):
        super(SupervisedVisionNetwork, self).__init__()
        self.tower1 = SimpleDenseNetwork(input_emb_dim, sup_hidden_layers, top_no_act = True)
        self.tower2 = SimpleDenseNetwork(input_emb_dim, sup_hidden_layers, top_no_act = True)
        self.tower3 = SimpleDenseNetwork(input_emb_dim, sup_hidden_layers, top_no_act = True)
        self.tower4 = SimpleDenseNetwork(input_emb_dim, sup_hidden_layers, top_no_act = True)
        self.tower5 = SimpleDenseNetwork(input_emb_dim, sup_hidden_layers, top_no_act = True)
        self.tower6 = SimpleDenseNetwork(input_emb_dim, sup_hidden_layers, top_no_act = True)
        
    def forward(self, x, user_type_onehot, idx_vec, reward_start, reward_range):
        tower1_out = self.tower1(x).unsqueeze(2)
        tower2_out = self.tower2(x).unsqueeze(2)
        tower3_out = self.tower3(x).unsqueeze(2)
        tower4_out = self.tower4(x).unsqueeze(2)
        tower5_out = self.tower5(x).unsqueeze(2)
        tower6_out = self.tower6(x).unsqueeze(2)
        user_flag = user_type_onehot.unsqueeze(1)
        user_tower_list = torch.concat([tower1_out,tower2_out,tower3_out,tower4_out,tower5_out,tower6_out], dim =2) #[bs,dim, 6]
        user_tower = torch.sigmoid(torch.mul(user_flag, user_tower_list).sum(dim = 2))  # 4 7
        # print('sup tower:', user_tower)
        # print(x,  user_type_onehot, idx_vec, reward_start, reward_range, user_flag)
        # print('------')
        # print(tower1_out,tower2_out,tower3_out,tower4_out,tower5_out,tower6_out)
        
        
        # pred_ratio = tf.gather(user_tower, idx_vec, batch_dims=1)
        pred_ratio = torch.gather(user_tower, 1, torch.LongTensor(idx_vec).cuda(device_controller.get_device())) # 4 2
        # 这个监督网络预测出来是一个ratio，根据ratio和label反解出时长。
        pred_reward = torch.Tensor(reward_start).cuda(device_controller.get_device()) + pred_ratio * torch.Tensor(reward_range).cuda(device_controller.get_device())
        return pred_ratio, pred_reward

class CriticNetwork(nn.Module):
    def __init__(self, input_emb_dim, critic_a_hidden_layers = [64, 32, 2]):
        super(CriticNetwork, self).__init__()
        self.ln = nn.LayerNorm(input_emb_dim)
        # 有俩supervised tower，一个是live 一个是 video。
        self.live_sup_tower = SupervisedVisionNetwork(input_emb_dim, sup_hidden_layers = [64, 32, 8])
        self.photo_sup_tower = SupervisedVisionNetwork(input_emb_dim, sup_hidden_layers = [64, 32, 8])
        
        # 用户tower
        self.tower1 = SimpleDenseNetwork(input_emb_dim, critic_a_hidden_layers, top_no_act = True)
        self.tower2 = SimpleDenseNetwork(input_emb_dim, critic_a_hidden_layers, top_no_act = True)
        self.tower3 = SimpleDenseNetwork(input_emb_dim, critic_a_hidden_layers, top_no_act = True)
        self.tower4 = SimpleDenseNetwork(input_emb_dim, critic_a_hidden_layers, top_no_act = True)
        self.tower5 = SimpleDenseNetwork(input_emb_dim, critic_a_hidden_layers, top_no_act = True)
        self.tower6 = SimpleDenseNetwork(input_emb_dim, critic_a_hidden_layers, top_no_act = True)
        
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
        pred_live_ratio, pred_live_time = self.live_sup_tower(base_tower, user_type_onehot, time_idx, live_start, live_range)
        pred_photo_ratio, pred_photo_time = self.photo_sup_tower(base_tower, user_type_onehot, photo_idx, photo_start, photo_range)
        pred_time_delta = pred_live_time - pred_photo_time
        # print('live photo:', pred_live_ratio, pred_live_time)
        # 通过两个预测的时间差计算reward
        pred_reward = torch.sigmoid(0.1 * pred_time_delta) #时间差越大越好，越小，越接近负数，不能给reward
        
        #用户tower
        tower1_out = self.tower1(base_tower).unsqueeze(2)
        tower2_out = self.tower2(base_tower).unsqueeze(2)
        tower3_out = self.tower3(base_tower).unsqueeze(2)
        tower4_out = self.tower4(base_tower).unsqueeze(2)
        tower5_out = self.tower5(base_tower).unsqueeze(2)
        tower6_out = self.tower6(base_tower).unsqueeze(2)
        user_flag = user_type_onehot.unsqueeze(1)
        a_user_tower = torch.concat([tower1_out,tower2_out,tower3_out,tower4_out,tower5_out,tower6_out], dim =2)     #[bs,dim, 6]
        a_user_tower = torch.mul(user_flag, a_user_tower).sum(dim = 2)
        #最终的Q值计算
        pred_q_value = pred_reward.detach() + 0.9 * F.elu(a_user_tower)
        # print('pred: ', pred_reward, a_user_tower)
        #reward（时间差的sigmoid）, 时间差, live时间比例，live反解时长，photo时间比例，photo反解时长，q值（reward+0.99*）
        return pred_reward, pred_time_delta, pred_live_ratio, pred_live_time, pred_photo_ratio, pred_photo_time, pred_q_value

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
        alpha_time = 1.0
        delta_time = 0.1
        alpha_photo = 2.0
        delta_photo = 0.1
        alpha = 1.0
        delta = 0.1
        
        cur_q1_pred_reward, cur_q1_pred_time_delta, cur_q1_pred_live_time_ratio, cur_q1_pred_live_time, cur_q1_pred_photo_time_ratio, cur_q1_pred_photo_time, cur_q1_value = self.critic1(cur_x, cur_labels, user_type_onehot)
        cur_q2_pred_reward, cur_q2_pred_time_delta, cur_q2_pred_live_time_ratio, cur_q2_pred_live_time, cur_q2_pred_photo_time_ratio, cur_q2_pred_photo_time, cur_q2_value = self.critic2(cur_x, cur_labels, user_type_onehot)
        
        cur_q_value = torch.min(cur_q1_value, cur_q2_value)
        cur_q_argmax = torch.argmax(cur_q_value, dim = 1)
        
        #先得到cur_action，如果(cur_labels)reco_ban ==1 cur_action={0.0 1.0} 否则=={1.0, 0.0}
        cur_action = self._get_cur_actions(cur_labels['reco_ban'])
        
        cur_q1_action_pred_reward = torch.sum(cur_q1_pred_reward * cur_action, dim=1, keepdim=True)
        cur_q2_action_pred_reward = torch.sum(cur_q2_pred_reward * cur_action, dim=1, keepdim=True)
        
        cur_q1_action_value = torch.sum(cur_q1_value * cur_action, dim=1, keepdim=True)
        cur_q2_action_value = torch.sum(cur_q2_value * cur_action, dim=1, keepdim=True)        
        
        cur_q1_action_pred_live_time_ratio = torch.sum(cur_q1_pred_live_time_ratio * cur_action, dim=1, keepdim=True)
        cur_q2_action_pred_live_time_ratio = torch.sum(cur_q2_pred_live_time_ratio * cur_action, dim=1, keepdim=True)
        
        cur_q1_action_pred_photo_time_ratio = torch.sum(cur_q1_pred_photo_time_ratio * cur_action, dim=1, keepdim=True)
        cur_q2_action_pred_photo_time_ratio = torch.sum(cur_q2_pred_photo_time_ratio * cur_action, dim=1, keepdim=True)
        
        nxt_q1_pred_reward, nxt_q1_pred_time_delta, nxt_q1_pred_live_time_ratio, nxt_q1_pred_live_time, nxt_q1_pred_photo_time_ratio, nxt_q1_pred_photo_time, nxt_q1_value = target_network.critic1(nxt_x, nxt_labels, user_type_onehot)
        
        nxt_q2_pred_reward, nxt_q2_pred_time_delta, nxt_q2_pred_live_time_ratio, nxt_q2_pred_live_time, nxt_q2_pred_photo_time_ratio, nxt_q2_pred_photo_time, nxt_q2_value = target_network.critic2(nxt_x, nxt_labels, user_type_onehot)
        
        nxt_q_value = torch.min(nxt_q1_value, nxt_q2_value)
        nxt_q_value, _ = torch.max(nxt_q_value, dim =1 , keepdim=True)
        cur_time_delta = torch.Tensor(cur_labels['time_delta']).reshape((-1, 1)).cuda(device_controller.get_device())
        cur_time_delta_reward = F.sigmoid(0.1 * cur_time_delta)
        q_label = cur_time_delta_reward + 0.9 * torch.Tensor(not_final).view(-1, 1).cuda(device_controller.get_device()) * nxt_q_value.detach()
        
        q1_live_time_reward_loss = new_huber_loss(torch.Tensor(cur_labels['time_ratio']).reshape((-1,1)).cuda(device_controller.get_device()), cur_q1_action_pred_live_time_ratio, 1.0, alpha_time, delta_time)
        q2_live_time_reward_loss = new_huber_loss(torch.Tensor(cur_labels['time_ratio']).reshape((-1,1)).cuda(device_controller.get_device()), cur_q2_action_pred_live_time_ratio, 1.0, alpha_time, delta_time)
        q1_photo_time_reward_loss = new_huber_loss(torch.Tensor(cur_labels['photo_ratio']).reshape((-1,1)).cuda(device_controller.get_device()), cur_q1_action_pred_photo_time_ratio, 1.0, alpha_photo, delta_photo)
        q2_photo_time_reward_loss = new_huber_loss(torch.Tensor(cur_labels['photo_ratio']).reshape((-1,1)).cuda(device_controller.get_device()), cur_q2_action_pred_photo_time_ratio, 1.0, alpha_photo, delta_photo)
        q1_reward_loss = new_huber_loss(cur_time_delta_reward, cur_q1_action_pred_reward, 1 * 1.0, alpha_time, delta_time)
        q2_reward_loss = new_huber_loss(cur_time_delta_reward, cur_q2_action_pred_reward, 1 * 1.0, alpha_time, delta_time)
        q1_loss = new_huber_loss(q_label, cur_q1_action_value, 1.0, alpha, delta)
        q2_loss = new_huber_loss(q_label, cur_q2_action_value, 1.0, alpha, delta)

        live_time_sup_loss = torch.sum(q1_live_time_reward_loss) + torch.sum(q2_live_time_reward_loss)
        photo_time_sup_loss = torch.sum(q1_photo_time_reward_loss) + torch.sum(q2_photo_time_reward_loss)
        time_delta_sup_loss = torch.sum(q1_reward_loss) + torch.sum(q2_reward_loss)
        sup_loss = live_time_sup_loss + photo_time_sup_loss + time_delta_sup_loss
        rl_loss = torch.sum(q1_loss) + torch.sum(q2_loss)

        return cur_q_value, live_time_sup_loss, photo_time_sup_loss, time_delta_sup_loss, sup_loss, rl_loss

class LiveRecoModelSGQCopy(nn.Module):
    # 目前效果最好的；
    def __init__(self, 
                 input_network = None,
                 base_output_dim = 128
                 ):
        super(LiveRecoModelSGQCopy, self).__init__()
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
        # print(not_final)
        cur_common_input_res = self.input_network(cur_features)
        
        user_type = cur_features['user_type'] # list: len = bs
        user_type_indices = torch.tensor(user_type, dtype=torch.long).cuda(device_controller.get_device()) - 1       # [1, 6]--->范围是0~5
        user_type_onehot = F.one_hot(user_type_indices, num_classes=6)  # [batch_size, user_type_num]
        cur_action_logits = self.actor_network(cur_common_input_res.detach(), user_type_onehot)
        cur_action_probs = F.softmax(cur_action_logits, dim=1)
        
        nxt_common_input_res = self.input_network(nxt_features)
        
        #交替训练 ----> 改成复制。
        time_q_value1, live_time_sup_loss1, photo_time_sup_loss1, time_delta_sup_loss1, sup_loss1, time_rl_loss1 = self.time_critic1(cur_common_input_res, nxt_common_input_res, cur_labels, nxt_labels, not_final, user_type_onehot, self.time_critic2)
        
        # 外面定义step，实现交替训练的逻辑
        # 再把源代码复制过来
        cur_action = self._get_cur_actions(cur_labels['reco_ban'])
        cur_time_reward =torch.Tensor(cur_labels['time_reward']).cuda(device_controller.get_device())
        
        time_q_value = time_q_value1

        sup_loss =  sup_loss1
        time_rl_loss = time_rl_loss1
        
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
