from abc import ABC
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions.categorical import Categorical
import numpy as np
from torch import from_numpy
from torch.optim.adam import Adam
# def conv_shape(input, kernel_size, stride, padding=0):
#     return (input + 2 * padding - kernel_size) // stride + 1


class QValueNetwork(nn.Module, ABC):
    def __init__(self, state_shape, n_actions):
        super(QValueNetwork, self).__init__()
        self.state_shape = state_shape
        self.n_actions = n_actions

        self.fc = nn.Linear(in_features=state_shape, out_features=512)
        self.q_value = nn.Linear(in_features=512, out_features=self.n_actions)

        nn.init.kaiming_normal_(self.fc.weight, nonlinearity="relu")
        self.fc.bias.data.zero_()
        nn.init.xavier_uniform_(self.q_value.weight)
        self.q_value.bias.data.zero_()

    def forward(self, states):
        x = states
        x = F.relu(self.fc(x))
        return self.q_value(x)


class PolicyNetwork(nn.Module, ABC):
    def __init__(self, state_shape, n_actions):
        super(PolicyNetwork, self).__init__()
        self.state_shape = state_shape
        self.n_actions = n_actions


        self.fc = nn.Linear(in_features=state_shape, out_features=512)
        self.logits = nn.Linear(in_features=512, out_features=self.n_actions)

        nn.init.kaiming_normal_(self.fc.weight, nonlinearity="relu")
        self.fc.bias.data.zero_()
        nn.init.xavier_uniform_(self.logits.weight)
        self.logits.bias.data.zero_()

    def forward(self, states):
        x = states
        x = F.relu(self.fc(x))
        logits = self.logits(x)
        probs = F.softmax(logits, -1)
        z = probs == 0.0
        z = z.float() * 1e-8
        return Categorical(probs), probs + z

class SAC(nn.Module):
    def __init__(self,
                input_network = None,
                base_output_dim = 128,
                device_name = 0 
                ):
        
        # self.config = config
        self.state_shape = base_output_dim
        self.n_actions = 2
        self.lr = 1e-3
        self.gamma = 0.9
        self.fixed_network_update_freq = 100
        # self.memory = Memory(memory_size=self.config["mem_size"])

        super(SAC, self).__init__()
        self.device = device_name
        self.input_network = input_network
        self.actor_network = PolicyNetwork(state_shape=self.state_shape, n_actions=self.n_actions).cuda(self.device)
        self.critic_network1 = QValueNetwork(state_shape=self.state_shape, n_actions=self.n_actions).cuda(self.device)
        self.critic_network2 = QValueNetwork(state_shape=self.state_shape, n_actions=self.n_actions).cuda(self.device)
        self.q_value_target_network1 = QValueNetwork(state_shape=self.state_shape,
                                                     n_actions=self.n_actions).cuda(self.device)
        self.q_value_target_network2 = QValueNetwork(state_shape=self.state_shape,
                                                     n_actions=self.n_actions).cuda(self.device)

        self.q_value_target_network1.load_state_dict(self.critic_network1.state_dict())
        self.q_value_target_network1.eval()

        self.q_value_target_network2.load_state_dict(self.critic_network2.state_dict())
        self.q_value_target_network2.eval()

        self.entropy_target = 0.98 * (-np.log(1 / self.n_actions))
        #self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        #self.alpha = self.log_alpha.exp()
        self.alpha = 1.0
        # self.q_value1_opt = Adam(self.critic_network1.parameters(), lr=self.lr)
        # self.q_value2_opt = Adam(self.critic_network2.parameters(), lr=self.lr)
        # self.policy_opt = Adam(self.actor_network.parameters(), lr=self.lr)
        #self.alpha_opt = Adam([self.log_alpha], lr=self.lr)

        self.update_counter = 0

    def forward(self, input):
       
        cur_features, nxt_features, cur_labels, nxt_labels, not_final  = input['cur_features'],input['nxt_features'],input['cur_labels'],input['nxt_labels'], input['not_final']
        # print(not_final)
        state = cur_features
        next_state = nxt_features
        actions = torch.tensor(cur_labels['reco_ban'],dtype=torch.int64).unsqueeze(1).cuda(self.device)
        rewards = torch.tensor(cur_labels['time_delta']).unsqueeze(1).cuda(self.device)
        rewards = F.sigmoid(0.1 * rewards)
        dones = torch.tensor(1-np.array(not_final)).unsqueeze(1).cuda(self.device)
        if 1 == 1:
            # Calculating the Q-Value target
            with torch.no_grad():
                next_states = self.input_network(next_state)
                _, next_probs = self.actor_network(next_states)
                next_log_probs = torch.log(next_probs)
                next_q1 = self.q_value_target_network1(next_states)
                next_q2 = self.q_value_target_network2(next_states)
                next_q = torch.min(next_q1, next_q2)
                next_v = (next_probs * (next_q - self.alpha * next_log_probs)).sum(-1).unsqueeze(-1)
                target_q = rewards + self.gamma * (~dones) * next_v
            states = self.input_network(state)
            q1 = self.critic_network1(states).gather(1, actions)
            q2 = self.critic_network2(states).gather(1, actions)
            q1_loss = F.mse_loss(q1, target_q,reduction='sum')
            q2_loss = F.mse_loss(q2, target_q,reduction='sum')

            # Calculating the Policy target
            _, probs = self.actor_network(states)
            log_probs = torch.log(probs)
            with torch.no_grad():
                q1 = self.critic_network1(states)
                q2 = self.critic_network2(states)
                q = torch.min(q1, q2)

            #policy_loss = (probs * (self.alpha.detach() * log_probs - q)).sum(-1).sum()
            policy_loss = (probs * (self.alpha * log_probs - q)).sum(-1).sum()
            # self.q_value1_opt.zero_grad()
            # q1_loss.backward(retain_graph=True)
            # self.q_value1_opt.step()
            # # print("q1_loss",q1_loss)
            # self.q_value2_opt.zero_grad()
            # q2_loss.backward(retain_graph=True)
            # self.q_value2_opt.step()
            # # print("q2_loss",q2_loss)
            # self.policy_opt.zero_grad()
            # policy_loss.backward()
            # self.policy_opt.step()
            # # print("policy_loss",policy_loss)
            log_probs = (probs * log_probs).sum(-1)
            #self.alpha_loss = -(self.log_alpha * (log_probs.detach() + self.entropy_target)).sum()

            # self.alpha_opt.zero_grad()
            # alpha_loss.backward()
            # self.alpha_opt.step()

            # self.update_counter += 1

            # self.alpha = self.log_alpha.exp()

            # # if self.update_counter % self.config["fixed_network_update_freq"] == 0:
            # if self.update_counter % self.fixed_network_update_freq == 0:
            #     self.hard_update_target_network()

            # return self.alpha_loss.item(), 0.5 * (q1_loss + q2_loss).item(), policy_loss.item()
            return probs, 0.5 * (q1_loss + q2_loss), policy_loss
            #return probs, q1_loss + q2_loss, policy_loss
    def update_network(self,total_loss,optimizer):
        # self.q_value1_opt.zero_grad()
        # q1_loss.backward(retain_graph=True)
        # self.q_value1_opt.step()
        # # print("q1_loss",q1_loss)
        # self.q_value2_opt.zero_grad()
        # q2_loss.backward(retain_graph=True)
        # self.q_value2_opt.step()
        # # print("q2_loss",q2_loss)
        # self.policy_opt.zero_grad()
        # policy_loss.backward()
        # self.policy_opt.step()
        # # print("policy_loss",policy_loss)
        # log_probs = (probs * log_probs).sum(-1)
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        #self.alpha_opt.zero_grad()
        #self.alpha_loss.backward()
        #self.alpha_opt.step()

        self.update_counter += 1

        #self.alpha = self.log_alpha.exp()

        # if self.update_counter % self.config["fixed_network_update_freq"] == 0:
        if self.update_counter % self.fixed_network_update_freq == 0:
            self.hard_update_target_network()
        
    # def choose_action(self, states, do_greedy=False):
    #     states = np.expand_dims(states, axis=0)
    #     states = from_numpy(states).byte().to(self.device)
    #     with torch.no_grad():
    #         #TODO
    #         dist, p = self.actor_network(states)
    #         if do_greedy:
    #             action = p.argmax(-1)
    #         else:
    #             action = dist.sample()
    #     return action.detach().cpu().numpy()[0]

    def hard_update_target_network(self):
        self.q_value_target_network1.load_state_dict(self.critic_network1.state_dict())
        self.q_value_target_network1.eval()
        self.q_value_target_network2.load_state_dict(self.critic_network2.state_dict())
        self.q_value_target_network2.eval()

    def set_to_eval_mode(self):
        self.actor_network.eval()
