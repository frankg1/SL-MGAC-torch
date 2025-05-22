import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DiscreteActor(nn.Module):
    def __init__(self, state_dim, action_dim, num_actions):
        super(DiscreteActor, self).__init__()
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 256)
        
        self.net = nn.Sequential(*[self.l1, nn.ReLU(inplace=True), self.l2, nn.ReLU(inplace=True), self.l3, nn.ReLU(inplace=True)])
        self.logits_layer = nn.Linear(256, num_actions)

    def forward(self, state):
        net_out = self.net(state)
        logits = self.logits_layer(net_out)
        return F.softmax(logits, dim=-1)
    
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(state_dim, 256)
        #self.l2 = nn.Linear(256 + action_dim, 256)
        #self.l3 = nn.Linear(256, 1)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)

    def forward(self, state, action):
        #q = F.relu(self.l1(state))
        #q = F.relu(self.l2(torch.cat([q, action], 1)))
        #return self.l3(q)
        q = F.relu(self.l1(state))
        q = F.relu(self.l2(q))
        q = self.l3(q)
        return torch.sum(q * action ,-1).unsqueeze(1)

class V_net(nn.Module):
    def __init__(self, state_dim):
        super(V_net, self).__init__()
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.l3(a)

class IQL(nn.Module):
    def __init__(self,
                input_network = None,
                base_output_dim = 128,
                device_name = 0,
                ):
        super(IQL, self).__init__()
        self.num_actions = 2#num_actions
        self.action_dim = 2
        self.state_dim = base_output_dim
        self.lr = 0.00005
        
        self.device = device_name
        self.input_network = input_network.cuda(self.device)
        self.actor_network = DiscreteActor(self.state_dim, self.action_dim, self.num_actions).cuda(self.device)
        self.actor_target = copy.deepcopy(self.actor_network)
        # self.actor_optimizer = torch.optim.Adam(self.actor_network.parameters(), lr=0.1 * self.lr)

        self.critic_network1 = Critic(self.state_dim, self.action_dim).cuda(self.device)
        self.critic_target = copy.deepcopy(self.critic_network1)
        # self.critic_optimizer = torch.optim.Adam(self.critic_network.parameters(), lr=self.lr, weight_decay=1e-2)

        self.critic_network2 = V_net(self.state_dim).cuda(self.device)
        self.V_optimizer = torch.optim.Adam(self.critic_network2.parameters(), lr=self.lr)

        self.discount = 0.9#discount
        self.tau = 0.005#tau
        self.expectile = 0.8#args.expectile
        # self.args = args
        

    # def select_action(self, state, deterministic=True):
    #     state = torch.FloatTensor(state.reshape(1, -1)).to(device)
    #     # state = norm_state(state)
    #     with torch.no_grad():
    #         probs = self.actor_network(state)
    #         if deterministic:
    #             action = torch.argmax(probs, dim=-1)
    #         else:
    #             action = torch.multinomial(probs, 1).squeeze()
    #         return action.item()

    def forward(self, input):
        cur_features, nxt_features, cur_labels, nxt_labels, not_final  = input['cur_features'],input['nxt_features'],input['cur_labels'],input['nxt_labels'], input['not_final']
        # print(not_final)
        state = cur_features
        next_state = nxt_features
        state = self.input_network(cur_features)
        next_state = self.input_network(nxt_features)
        action = torch.tensor(cur_labels['reco_ban'],dtype=torch.int64).unsqueeze(1).cuda(self.device)
        reward = torch.tensor(cur_labels['time_delta']).unsqueeze(1).cuda(self.device)
        reward = F.sigmoid(0.1 * reward)
        done = torch.tensor(1-np.array(not_final)).unsqueeze(1).cuda(self.device)
        not_done = torch.tensor(not_final).unsqueeze(1).cuda(self.device)
        #import pdb
        #pdb.set_trace()
        action_one_hot = F.one_hot(action, num_classes=self.num_actions).float().squeeze()
        target_Q = self.critic_target(state, action_one_hot).detach()
        predict_V = self.critic_network2(state)
        u = target_Q - predict_V  # (batch, 1)
        self.V_loss = (torch.abs(self.expectile - torch.le(u, 0).float()) * (u ** 2)).sum()

        target_Q = reward + (not_done * self.discount * self.critic_network2(next_state)).detach()
        action_one_hot = F.one_hot(action, num_classes=self.num_actions).float().squeeze()
        current_Q = self.critic_network1(state, action_one_hot)
        self.critic_loss = F.mse_loss(target_Q, current_Q,reduction='sum')

        action_probs = self.actor_network(state)
        action_one_hot = F.one_hot(action, num_classes=self.num_actions).float().squeeze()
        # print(action_probs.shape,action_one_hot.shape)
        # print(action_probs,action_one_hot)
        Q = self.critic_network1(state, action_one_hot)
        V = self.critic_network2(state)
        #actor_loss = (torch.exp(Q - V) * torch.log(action_probs + 1e-8)).mean()
        actor_loss = (0.1 * torch.exp(Q - V).detach() * torch.log(torch.sum(action_probs*action_one_hot,-1).unsqueeze(1)+1e-8)).sum()
        #return action_probs,self.critic_loss + self.V_loss, actor_loss
        return action_probs, self.critic_loss, actor_loss
    
    def update_network(self,total_loss,optimizer):
        # self.V_optimizer.zero_grad()
        # V_loss.backward(retain_graph=True)
        # self.V_optimizer.step()

        # self.critic_optimizer.zero_grad()
        # critic_loss.backward(retain_graph=True)
        # self.critic_optimizer.step()

        # self.actor_optimizer.zero_grad()
        # actor_loss.backward()
        # self.actor_optimizer.step()
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        for param, target_param in zip(self.critic_network1.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        for param, target_param in zip(self.actor_network.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

