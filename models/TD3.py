import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# from utils import process_reward, norm_state

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, temperature=1.0):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)
        #self.l3 = nn.softmax(256, action_dim)
        #self.temperature = temperature

    def forward(self, state):
        x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))
        logits = self.l3(x)
        #x = F.relu(torch.nn.functional.linear (state, self.l1.weight.clone(), self.l1.bias))
        #x = F.relu(torch.nn.functional.linear (x, self.l2.weight.clone(), self.l2.bias))
        #logits = torch.nn.functional.linear (x, self.l3.weight.clone(), self.l3.bias)
        #probs = F.softmax(logits, -1)
        #z = probs == 0.0
        #z = z.float() * 1e-8
        return logits
        #return Categorical(probs), probs+z

    def sample_action(self, state):
        logits = self.forward(state)
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-20) + 1e-20)
        noisy_logits = (logits + gumbel_noise) / self.temperature
        soft_samples = F.softmax(noisy_logits, dim=-1)
        hard_samples = torch.zeros_like(soft_samples)
        hard_samples.scatter_(-1, torch.argmax(soft_samples, dim=-1, keepdim=True), 1.0)
        samples = soft_samples + (hard_samples - soft_samples).detach()
        return samples

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        # Q1 architecture
        #self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l1 = nn.Linear(state_dim,256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)#1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, action_dim)

    #def forward(self, state, action):
    def forward(self, state):
        #sa = torch.cat([state, action], 1)
        # print(sa.shape)
        q1 = F.relu(self.l1(state))
        q12 = F.relu(self.l2(q1))
        q13 = self.l3(q12)
        #q1 = F.relu(torch.nn.functional.linear (state, self.l1.weight.clone(), self.l1.bias))
        #q1  = F.relu(torch.nn.functional.linear (q1, self.l2.weight.clone(), self.l2.bias))
        #q1 = torch.nn.functional.linear (q1, self.l3.weight.clone(), self.l3.bias)

        q2 = F.relu(self.l4(state))
        q22 = F.relu(self.l5(q2))
        q23 = self.l6(q22)
        return q13, q23
    def Q1(self, state):
        #q1 = F.relu(torch.nn.functional.linear(state, self.l1.weight.clone(), self.l1.bias))
        #q1  = F.relu(torch.nn.functional.linear(q1, self.l2.weight.clone(), self.l2.bias))
        #q1 = torch.nn.functional.linear (q1, self.l3.weight.clone(), self.l3.bias)
        q1 = F.relu(self.l1(state))
        q12 = F.relu(self.l2(q1))
        q13 = self.l3(q12)
        return q13

class TD3(nn.Module):
    def __init__(self,
                input_network = None,
                base_output_dim = 128, 
                device_name=0
                ):
        super(TD3, self).__init__()
        self.state_dim = base_output_dim
        self.action_dim = 2
        self.lr = 1e-3
        self.discount = 0.9#discount
        self.tau = 0.005#tau
        # self.policy_noise = #policy_noise
        # self.noise_clip = noise_clip
        self.policy_freq = 2#policy_freq
        self.alpha = 1#1000#alpha
        self.total_it = 0

        self.device = device_name
        self.input_network = input_network.cuda(self.device)
        self.actor_network = Actor(self.state_dim , self.action_dim).cuda(self.device)
        self.actor_target = copy.deepcopy(self.actor_network)
        # self.actor_optimizer = torch.optim.Adam(self.actor_network.parameters(), lr=0.1 * self.lr)

        self.critic_network = Critic(self.state_dim, self.action_dim).cuda(self.device)
        self.critic_target = copy.deepcopy(self.critic_network)
        # self.critic_optimizer = torch.optim.Adam(self.critic_network.parameters(), lr=self.lr)


    def forward(self, input):
        self.total_it += 1
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
        with torch.no_grad():
            # Select action according to policy and add clipped noise
            logits_next = self.actor_target(next_state)
            #gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits_next) + 1e-20) + 1e-20)
            #noisy_logits_next = (logits_next + gumbel_noise) / 1.0  # Temperature set to 1.0
            #soft_samples_next = F.softmax(noisy_logits_next, dim=-1)
            #hard_samples_next = torch.zeros_like(soft_samples_next)
            #hard_samples_next.scatter_(-1, torch.argmax(soft_samples_next, dim=-1, keepdim=True), 1.0)
            #next_action = hard_samples_next
            gumbel_softmax = F.gumbel_softmax(logits_next, tau=1.0, hard=True)
            next_action = gumbel_softmax
            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.discount * target_Q.detach().gather(1,torch.argmax(next_action,1).unsqueeze(1))
        # Get current Q estimates
        current_Q1, current_Q2 = self.critic_network(state)

        # Compute critic_network loss
        self.critic_loss = 0.5 * (F.mse_loss(current_Q1.gather(1,action), target_Q,reduction='sum') + F.mse_loss(current_Q2.gather(1,action), target_Q,reduction='sum'))
        # # critic_loss.requires_grad_()
        # # Optimize the critic_network
        # self.critic_optimizer.zero_grad()
        # critic_loss.backward(retain_graph=True)
        # self.critic_optimizer.step()
        # Compute actor_network loss
        #state1 = self.input_network(cur_features).detach()
        #logits = self.actor_network(state1)
        logits = self.actor_network(state)
      
        #gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-20) + 1e-20)
        #noisy_logits = (logits + gumbel_noise) / 1.0  # Temperature set to 1.0
        #soft_samples = F.softmax(noisy_logits, dim=-1)
        #hard_samples = torch.zeros_like(soft_samples)
        #hard_samples = torch.scatter(hard_samples,-1, torch.argmax(soft_samples, dim=-1, keepdim=True), 1.0)
        #pi = hard_samples
        gumbel_softmax = F.gumbel_softmax(logits, tau=1.0, hard=True)
        pi = gumbel_softmax
        #Q = self.critic_network.Q1(state).gather(1,torch.argmax(pi,1).unsqueeze(1))
        #Q = self.critic_network.Q1(state)
        #state1 = self.input_network(cur_features).detach()
        #import pdb
        #pdb.set_trace()
        Q = self.critic_network.Q1(state).detach() 
        #Q = torch.sum(current_Q1*pi,-1).unsqueeze(1)
        Q = torch.sum(Q*pi,-1).unsqueeze(1)
        lmbda = self.alpha

        self.actor_loss = -lmbda * Q.sum()
        #self.actor_loss = -lmbda * torch.sum(Q * action, dim = 1, keepdim = True)
        #return logits, self.critic_loss, self.actor_loss
        return F.softmax(logits,-1), self.critic_loss, self.actor_loss

    def update_network(self,total_loss,optimizer):
        
        # optimizer.zero_grad()
        # total_loss.backward()
        # optimizer.step()
        optimizer.zero_grad()
        self.critic_loss.backward(retain_graph=True)
        # self.critic_loss.backward()
        if self.total_it % self.policy_freq == 0:
            #optimizer.zero_grad()
            self.actor_loss.backward()
            optimizer.step()
            # Update the frozen target models
            for param, target_param in zip(self.critic_network.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor_network.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        else:
            optimizer.step()

class TD3_BC(nn.Module):
    def __init__(self,
                input_network = None,
                base_output_dim = 128, 
                device_name=0
                ):
        super(TD3_BC, self).__init__()
        self.state_dim = base_output_dim
        self.action_dim = 2
        self.lr = 1e-3
        self.discount = 0.9#discount
        self.tau = 0.005#tau
        # self.policy_noise = #policy_noise
        # self.noise_clip = noise_clip
        self.policy_freq = 2#policy_freq
        self.alpha = 1#100#1e-3#alpha
        self.bc_rate = 0.001#0.1
        self.total_it = 0
        self.device = device_name
        self.input_network = input_network.cuda(self.device)
        self.actor_network = Actor(self.state_dim, self.action_dim).cuda(self.device)
        self.actor_target = copy.deepcopy(self.actor_network)
        # self.actor_optimizer = torch.optim.Adam(self.actor_network.parameters(), lr=0.1*self.lr)

        self.critic_network = Critic(self.state_dim, self.action_dim).cuda(self.device)
        self.critic_target = copy.deepcopy(self.critic_network)
        # self.critic_optimizer = torch.optim.Adam(self.critic_network.parameters(), lr=self.lr)

    # def select_action(self, state):
    #     state = torch.FloatTensor(state.reshape(1, -1)).to(device)
	# 	# state=norm_state(state)
    #     return self.actor_network(state)


    def forward(self, input):
        self.total_it += 1

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
        with torch.no_grad():
            # Select action according to policy and add clipped noise
            logits_next = self.actor_target(next_state)
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits_next) + 1e-20) + 1e-20)
            noisy_logits_next = (logits_next + gumbel_noise) / 1.0  # Temperature set to 1.0
            soft_samples_next = F.softmax(noisy_logits_next, dim=-1)
            hard_samples_next = torch.zeros_like(soft_samples_next)
            hard_samples_next.scatter_(-1, torch.argmax(soft_samples_next, dim=-1, keepdim=True), 1.0)
            next_action = hard_samples_next

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.discount * target_Q.detach().gather(1,torch.argmax(next_action,1).unsqueeze(1))

		# Get current Q estimates
        current_Q1, current_Q2 = self.critic_network(state)

		# Compute critic_network loss
        self.critic_loss = 0.5 * (F.mse_loss(current_Q1.gather(1,action), target_Q,reduction='sum') + F.mse_loss(current_Q2.gather(1,action), target_Q,reduction='sum'))

        # Compute actor_network loss
        
        logits = self.actor_network(state)
        #gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-20) + 1e-20)
        #noisy_logits = (logits + gumbel_noise) / 1.0  # Temperature set to 1.0
        #soft_samples = F.softmax(noisy_logits, dim=-1)
        #hard_samples = torch.zeros_like(soft_samples)
        #hard_samples.scatter_(-1, torch.argmax(soft_samples, dim=-1, keepdim=True), 1.0)
        #pi = hard_samples
        pi = F.gumbel_softmax(logits, tau=1.0, hard=True)
        #import pdb
        #pdb.set_trace()
        Q = self.critic_network.Q1(state).detach()
        #Q = Q.gather(1,torch.argmax(pi,1).unsqueeze(1))
        Q = torch.sum(Q*pi,-1).unsqueeze(1)
        lmbda = self.alpha/Q.abs().mean()
	
        self.bc_loss=F.mse_loss(pi.to(dtype=torch.float32), F.one_hot(action,num_classes=2).to(dtype=torch.float32),reduction='sum')#F.mse_loss(torch.argmax(pi,1).unsqueeze(1), action) 
        self.actor_loss = -lmbda * Q.sum() + self.bc_loss*self.bc_rate
        # critic_loss.requires_grad_()
        #return logits, self.critic_loss, self.actor_loss
        return F.softmax(logits,-1), self.critic_loss, self.actor_loss
    def update_network(self,total_loss,optimizer):
        optimizer.zero_grad()
        self.critic_loss.backward(retain_graph=True)
        #optimizer.step()
        # print(critic_loss)
		# Delayed policy updates
        if self.total_it % self.policy_freq == 0:
			
            #optimizer.zero_grad()
            self.actor_loss.backward()
            optimizer.step()

			# Update the frozen target models
            for param, target_param in zip(self.critic_network.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor_network.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        else:
            optimizer.step()
