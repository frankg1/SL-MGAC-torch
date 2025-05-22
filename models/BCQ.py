import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
class FC_Q(nn.Module):
	def __init__(self, state_dim, num_actions):
		super(FC_Q, self).__init__()
		# print(state_dim)
		self.q1 = nn.Linear(state_dim, 256)
		self.q2 = nn.Linear(256, 256)
		self.q3 = nn.Linear(256, num_actions)

		self.i1 = nn.Linear(state_dim, 256)
		self.i2 = nn.Linear(256, 256)
		self.i3 = nn.Linear(256, num_actions)


	def forward(self, state):
		q = F.relu(self.q1(state))
		q = F.relu(self.q2(q))

		i = F.relu(self.i1(state))
		i = F.relu(self.i2(i))
		i = F.relu(self.i3(i))
		return self.q3(q), F.log_softmax(i, dim=1), i


class discrete_BCQ(nn.Module):
	def __init__(
		self,
		input_network = None,
        base_output_dim = 128,
		num_actions=2,
		BCQ_threshold=0.3,
		discount=0.9,
		optimizer="Adam",
		optimizer_parameters={},
		polyak_target_update=True,
		target_update_frequency=8e3,
		tau=0.005,
		initial_eps = 1,
		end_eps = 0.001,
		eps_decay_period = 25e4,
		eval_eps=0.001,
		device_name = 0,
		
		):
		super(discrete_BCQ, self).__init__()
		self.device = device_name
		self.input_network = input_network

		# Determine network type
		# print(state_dim)
		self.critic_network = FC_Q(base_output_dim, num_actions).cuda(self.device)
		self.Q_target = copy.deepcopy(self.critic_network)
		# self.Q_optimizer = getattr(torch.optim, optimizer)(self.critic_network.parameters(), **optimizer_parameters)

		self.discount = discount

		# Target update rule
		self.maybe_update_target = self.polyak_target_update if polyak_target_update else self.copy_target_update
		self.target_update_frequency = target_update_frequency
		self.tau = tau

		# Decay for eps
		self.initial_eps = initial_eps
		self.end_eps = end_eps
		self.slope = (self.end_eps - self.initial_eps) / eps_decay_period

		# Evaluation hyper-parameters
		self.state_shape = (-1, base_output_dim) ### need to pass framesize
		self.eval_eps = eval_eps
		self.num_actions = num_actions

		# Threshold for "unlikely" actions
		self.threshold = BCQ_threshold

		# Number of training iterations
		self.iterations = 0

	def forward(self, input):
		
		cur_features, nxt_features, cur_labels, nxt_labels, not_final  = input['cur_features'],input['nxt_features'],input['cur_labels'],input['nxt_labels'], input['not_final']
		state = cur_features
		next_state = nxt_features
		action = torch.tensor(cur_labels['reco_ban'],dtype=torch.int64).unsqueeze(1).cuda(self.device)
		reward = torch.tensor(cur_labels['time_delta']).unsqueeze(1).cuda(self.device)
		reward = F.sigmoid(0.1 * reward)
		done = torch.tensor(1-np.array(not_final)).unsqueeze(1).cuda(self.device)
        
		with torch.no_grad():
			nxt_common_input_res = self.input_network(next_state)
			q, imt, i = self.critic_network(nxt_common_input_res)
			imt = imt.exp()
			imt = (imt/imt.max(1, keepdim=True)[0] > self.threshold).float()

			# Use large negative number to mask actions from argmax
			next_action = (imt * q + (1 - imt) * -1e8).argmax(1, keepdim=True)
			#print(f'next_action {next_action}')

			q, imt, i = self.Q_target(nxt_common_input_res)
			# print(reward)
			# print(done)
			# print(((q.gather(1, next_action).reshape(-1, 1))).shape)
			target_Q = reward + done * self.discount * q.gather(1, next_action).reshape(-1, 1)

		# Get current Q estimate
		cur_common_input_res = self.input_network(state)
		current_Q, imt, i = self.critic_network(cur_common_input_res)
		# print(current_Q)
		# print(action)
		current_Q1 = current_Q.gather(1, action)

		# Compute Q loss
		q_loss = F.smooth_l1_loss(current_Q1, target_Q,reduction='sum')
		i_loss = F.nll_loss(imt, action.reshape(-1),reduction='sum')

		Q_loss = q_loss + i_loss + 1e-2 * i.pow(2).mean(dim=1).sum()
		'''拆出去'''
		# # print("Q_loss:",Q_loss)
		# # Optimize the Q
		# self.Q_optimizer.zero_grad()
		# Q_loss.backward()
		# self.Q_optimizer.step()

		# # Update target network by polyak or full copy every X iterations.
		# self.iterations += 1
		# self.maybe_update_target()
		return current_Q , Q_loss
	
	def update_network(self,Q_loss,optimizer):
        # print("Q_loss:",Q_loss)
		# Optimize the Q
		# self.Q_optimizer.zero_grad()
		optimizer.zero_grad()
		Q_loss.backward()
		# self.Q_optimizer.step()
		optimizer.step()

		# Update target network by polyak or full copy every X iterations.
		self.iterations += 1
		self.maybe_update_target()


	def polyak_target_update(self):
		for param, target_param in zip(self.critic_network.parameters(), self.Q_target.parameters()):
		   target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

	def copy_target_update(self):
		if self.iterations % self.target_update_frequency == 0:
			 self.Q_target.load_state_dict(self.critic_network.state_dict())
