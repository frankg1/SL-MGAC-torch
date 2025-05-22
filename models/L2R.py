import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .Base import SimpleDenseNetwork
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        pred_ratio = torch.gather(user_tower, 1, torch.LongTensor(idx_vec)) # 4 2
        # 这个监督网络预测出来是一个ratio，根据ratio和label反解出时长。
        pred_reward = torch.Tensor(reward_start) + pred_ratio * torch.Tensor(reward_range)
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
        pred_q_value = pred_reward.detach() + 0.9 * F.relu(a_user_tower)
        # print('pred: ', pred_reward, a_user_tower)
        #reward（时间差的sigmoid）, 时间差, live时间比例，live反解时长，photo时间比例，photo反解时长，q值（reward+0.99*）
        

        return pred_reward, pred_time_delta, pred_live_ratio, pred_live_time, pred_photo_ratio, pred_photo_time, pred_q_value

def new_huber_loss(label, pred, weight, alpha_val, delta_val):
    residual = torch.abs(label - pred)
    min_residual = torch.minimum(residual, torch.tensor(delta_val))
    max_neg_pred = torch.maximum(-pred, torch.tensor(0.0))    
    huber_loss = weight * (0.5 * torch.square(min_residual) + alpha_val * (residual - min_residual + max_neg_pred))
    return huber_loss

class StepCounter:
    def __init__(self):
        self.step = 0
    def get_step(self):
        return self.step
    def increment_step(self):
        self.step += 1
step_counter = StepCounter()

class L2R(nn.Module):
    def __init__(self, 
                 input_network = None,
                 base_output_dim = 128,
                ):
        super(L2R, self).__init__()
        self.device = device
        self.input_network = input_network.to(device)
        self.critic_network = CriticNetwork(base_output_dim)
        self.critic_network2 = CriticNetwork(base_output_dim)
    def _get_cur_actions(self, reco_ban):
        actions = []
        for i in reco_ban:
            if i == 0:
                actions.append([1.0, 0.0])
            else:
                actions.append([0.0, 1.0])
        return torch.Tensor(actions)
    
    def forward(self, input):
        cur_features, nxt_features, cur_labels, nxt_labels, not_final  = input['cur_features'],input['nxt_features'],input['cur_labels'],input['nxt_labels'], input['not_final']
        # print(not_final)
        cur_common_input_res = self.input_network(cur_features)
        
        user_type = cur_features['user_type'] # list: len = bs
        user_type_indices = torch.tensor(user_type, dtype=torch.long) - 1       # [1, 6]--->范围是0~5
        user_type_onehot = F.one_hot(user_type_indices, num_classes=6)  # [batch_size, user_type_num]
        # cur_action_logits = self.actor_network(cur_common_input_res.detach(), user_type_onehot)
        # cur_action_probs = F.softmax(cur_action_logits, dim=1)
        
        nxt_common_input_res = self.input_network(nxt_features)
        
        #交替训练 ----> 改成复制。
        pred_reward, pred_time_delta, pred_live_ratio, pred_live_time, pred_photo_ratio, pred_photo_time, pred_q_value = self.critic_network(cur_common_input_res, cur_labels, user_type_onehot)
        
        pred_reward2, pred_time_delta2, pred_live_ratio2, pred_live_time2, pred_photo_ratio2, pred_photo_time2, pred_q_value2 = self.critic_network2(cur_common_input_res, cur_labels, user_type_onehot)
        
        cur_reward = torch.tensor(cur_labels['time_delta']).unsqueeze(1)
        cur_action = torch.tensor(cur_labels['reco_ban'],dtype=torch.int64).unsqueeze(1)
        # print(cur_reward.shape,cur_action.shape)
        action_pred_reward = torch.sum(pred_reward * cur_action, dim=1, keepdims=True) 
        reward_loss = new_huber_loss(cur_reward, action_pred_reward, 1.0, 100.0, 0.1)

        action_pred_reward2 = torch.sum(pred_reward2 * cur_action, dim=1, keepdims=True) 
        reward_loss2 = new_huber_loss(cur_reward, action_pred_reward2, 1.0, 100.0, 0.1)

        # print(action_pred_reward.shape,action_pred_reward2.shape)
        reward_loss = torch.sum(reward_loss)
        reward_loss2 = torch.sum(reward_loss2)
        # 外面定义step，实现交替训练的逻辑
        # 再把源代码复制过来
        cur_action = self._get_cur_actions(cur_labels['reco_ban'])
        cur_time_reward =torch.Tensor(cur_labels['time_reward']) 
        
        step_mod = step_counter.get_step() % 2
        
        time_q_value = (1.0 - step_mod) * pred_reward + step_mod * pred_reward2
        # live_time_sup_loss = (1.0 - step_mod) * live_time_sup_loss1 + step_mod * live_time_sup_loss2
        # photo_time_sup_loss = (1.0 - step_mod) * photo_time_sup_loss1 + step_mod * photo_time_sup_loss2
        # time_delta_sup_loss = (1.0 - step_mod) * time_delta_sup_loss1 + step_mod * time_delta_sup_loss2
        # sup_loss = (1.0 - step_mod) * sup_loss1 + step_mod * sup_loss2
        time_rl_loss = (1.0 - step_mod) * reward_loss + step_mod * reward_loss2
        # print('time_rl_loss: ', time_rl_loss1.item(), time_rl_loss2.item())
        
        # cur_action_prob = torch.sum(cur_action_probs * cur_action, dim=1, keepdim=True)
        # cur_log_action_prob = torch.log(cur_action_prob + 1e-10)
        
        time_actor_weight = 10.0
        ctr_loss_weight = 100.0
        cur_reg_weight = 0.0
        
        tot_q_value = time_actor_weight * time_q_value
        tot_q_probs = torch.softmax(tot_q_value.detach(), dim=1)
        tot_q_action_prob = torch.sum(tot_q_probs * cur_action, dim=1, keepdim=True)
        ones = torch.ones_like(tot_q_action_prob)
        # ce_loss = torch.sum(ctr_loss_weight * F.cross_entropy(cur_action_logits, tot_q_probs.argmax(dim=1), reduction='none').view(-1, 1))
        # reg_loss = torch.sum(cur_reg_weight * torch.abs(cur_action_prob - 0.5))
        # actor_loss = ce_loss + reg_loss
        rl_loss = time_rl_loss
        # critic_loss = sup_loss + rl_loss
        critic_loss = rl_loss
        
        # return cur_action, cur_time_reward, time_rl_loss, critic_loss
        return critic_loss
    
    def update_network(self,Q_loss,optimizer):
        optimizer.zero_grad()
        Q_loss.backward()
        optimizer.step()
        step_counter.increment_step()
