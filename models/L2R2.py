import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Used for Box2D / Toy problems
class FC_Q(nn.Module):
	def __init__(self, state_dim, num_actions):
		super(FC_Q, self).__init__()
		self.l1 = nn.Linear(state_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, num_actions)


	def forward(self, state):
		q = F.relu(self.l1(state))
		q = F.relu(self.l2(q))
		return self.l3(q)


class L2R2(nn.Module):
	def __init__(
		self,
		input_network = None,
        base_output_dim = 128,
		is_atari=False,
		num_actions=2,
		state_dim=128,
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
		device_name=0
	):

		super(L2R2, self).__init__()
		self.device = device_name
		
		self.input_network = input_network.cuda(self.device)
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
		self.state_shape = (-1, 4, 84, 84) if is_atari else (-1, base_output_dim) ### need to pass framesize
		self.eval_eps = eval_eps
		self.num_actions = num_actions

		# Number of training iterations
		self.iterations = 0


	def forward(self, input):
		
		cur_features, nxt_features, cur_labels, nxt_labels, not_final  = input['cur_features'],input['nxt_features'],input['cur_labels'],input['nxt_labels'], input['not_final']
        # print(not_final)
		state = cur_features
		#next_state = nxt_features
		action = torch.tensor(cur_labels['reco_ban'],dtype=torch.int64).unsqueeze(1).cuda(self.device)
		reward = torch.tensor(cur_labels['time_delta']).unsqueeze(1).cuda(self.device)
		reward = F.sigmoid(0.1 * reward)
		#done = torch.tensor(1-np.array(not_final)).unsqueeze(1).to(device)
		
		# Compute the target Q value
		#with torch.no_grad():
		#	nxt_embedding = self.input_network(next_state)
		#	target_Q = reward + done * self.discount * self.Q_target(nxt_embedding).max(1, keepdim=True)[0]

		# Get current Q estimate
		cur_embedding = self.input_network(state)
		current_Q = self.critic_network(cur_embedding)
		current_pred_reward = current_Q.gather(1, action)

		# Compute Q loss
		#Q_loss = F.smooth_l1_loss(current_Q, target_Q)
		Q_loss = F.smooth_l1_loss(current_pred_reward,reward,reduction='sum')
		return current_Q, Q_loss
		# print(Q_loss)
		# Optimize the Q
		# self.Q_optimizer.zero_grad()
		# Q_loss.backward()
		# self.Q_optimizer.step()

		# # Update target network by polyak or full copy every X iterations.
		# self.iterations += 1
		# self.maybe_update_target()
	def update_network(self, Q_loss,optimizer):
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


	# def save(self, filename):
	# 	torch.save(self.critic_network.state_dict(), filename + "_Q")
	# 	torch.save(self.Q_optimizer.state_dict(), filename + "_optimizer")


	# def load(self, filename):
	# 	self.critic_network.load_state_dict(torch.load(filename + "_Q"))
	# 	self.Q_target = copy.deepcopy(self.critic_network)
	# 	self.Q_optimizer.load_state_dict(torch.load(filename + "_optimizer"))
