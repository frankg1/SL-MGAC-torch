from dataloaders.MyDataloader import LiveRecommendationDataset
import torch
import json
import numpy as np
import pandas as pd
import torch.nn.functional as F
from torch.distributions.normal import Normal
from models.Base import device_controller
from models.Base import InputNetwork
################ import all the models
from models.MyModel import LiveRecoModel, step_counter
from models.MyModelK1 import LiveRecoModelK1
from models.MyModelK2 import LiveRecoModelK2
from models.MyModelK3 import LiveRecoModelK3
from models.MyModelK4 import LiveRecoModelK4
from models.MyModelK5 import LiveRecoModelK5

from models.Model_Gr import LiveRecoModelGr
from models.Model_GrLoss import LiveRecoModelGrLoss
from models.Model_GrBu import LiveRecoModelGrBu
from models.Model_GrBuSl import LiveRecoModelGrBuSl
from models.Model_GrBuSl_Copy import LiveRecoModelGrBuSlCopy

from models.Model_LN import LiveRecoModelLN
from models.Model_SG import LiveRecoModelSG
from models.Model_SQ import LiveRecoModelSQ
from models.Model_SGQ import LiveRecoModelSGQ
from models.Model_NotQ import LiveRecoModelNotQ
from models.BCQ import discrete_BCQ
from models.DQN import DQN
#from models.L2R import L2R
from models.L2R2 import L2R2
from models.SAC import SAC
from models.TD3 import TD3
from models.TD3 import TD3_BC
from models.IQL import IQL
################
from time import time
import argparse



parser = argparse.ArgumentParser('desc')
parser.add_argument("--txt_file_path", type=str, default = 'rl_model_train_data_for_torch_1011_cleaned_test.txt', required = False)
parser.add_argument("--model_name", type=str, default='sl_mgac',
                    choices=["Model","ModelK1","ModelK2","ModelK3","ModelK4","ModelK5","Model_Gr",
                             "Model_GrBu","Model_GrBuSl","Model_LN","Model_SG","Model_SQ","Model_SGQ",
                             "Model_NotQ", "Model_SGSup", "Model_GrBuSlCopy","Model_GrLoss"], required=True)
parser.add_argument("--device_name", type=int, default=0, required=False)
parser.add_argument("--model_load_dir", type=str, default='save_weights/', required=False)
parser.add_argument("--epoch", type=int, default=50, required=False)
args = parser.parse_args()
device_controller.set_device(args.device_name)
##########
# mapping
# 新增加模型
# 1、需要在上面import
# 2、choices 增加model name
# 3、mapping中做好映射
model_mapping = {
    "Model": LiveRecoModel,
    "ModelK1": LiveRecoModelK1,
    "ModelK2": LiveRecoModelK2,
    "ModelK3": LiveRecoModelK3,
    "ModelK4": LiveRecoModelK4,
    "ModelK5": LiveRecoModelK5,
    "Model_Gr": LiveRecoModelGr,
    "Model_GrLoss": LiveRecoModelGrLoss,
    "Model_GrBu": LiveRecoModelGrBu,
    "Model_GrBuSl": LiveRecoModelGrBuSl,
    "Model_GrBuSlCopy": LiveRecoModelGrBuSlCopy,
    "Model_LN": LiveRecoModelLN,
    "Model_SG": LiveRecoModelSG,
    "Model_SQ": LiveRecoModelSQ,
    "Model_SGQ": LiveRecoModelSGQ,
    "Model_NotQ": LiveRecoModelNotQ,
    "BCQ": discrete_BCQ,
    "DQN": DQN,
    "L2R": L2R2,
    "SAC": SAC,
    "TD3": TD3,
    "TD3_BC": TD3_BC,
    "IQL": IQL
}
alabtion = True if args.model_name in ["Model","ModelK1","ModelK2","ModelK3","ModelK4","ModelK5",
                                       "Model_Gr","Model_GrBu","Model_GrBuSl",
                                       "Model_LN","Model_SG","Model_SQ","Model_SGQ",
                                       "Model_NotQ", "Model_SGSup","Model_GrBuSlCopy","Model_GrLoss"] else False
valueBased = True if args.model_name in ["BCQ", "DQN", "L2R"] else False
########
# input network定义，不要动！
embedding_dim=32
hash_size=5000
hidden_layers=[256,128]
input_network = InputNetwork(embedding_dim, hash_size, hidden_layers=hidden_layers)
input_network = input_network.cuda(args.device_name)
#########
# model定义 
live_model_class = model_mapping.get(args.model_name)
if live_model_class is None:
    raise ValueError(f"Unknown model name {args.model_name}")
if not alabtion:
    live_model = live_model_class(input_network=input_network, base_output_dim=128)
    try:
        live_model.critic_network.load_state_dict(torch.load(args.model_load_dir + f'{args.model_name}_critic_network.pth'))
    except:
        live_model.critic_network1.load_state_dict(torch.load(args.model_load_dir + f'{args.model_name}_critic_network1.pth'))
        live_model.critic_network2.load_state_dict(torch.load(args.model_load_dir + f'{args.model_name}_critic_network2.pth'))
    if not valueBased:
        live_model.actor_network.load_state_dict(torch.load(args.model_load_dir + f'{args.model_name}_actor_network.pth'))
else:
    live_model = live_model_class(input_network=input_network, base_output_dim=128)
    live_model.actor_network.load_state_dict(torch.load(args.model_load_dir + f'{args.model_name}_actor_network_{args.epoch}.pth', map_location='cuda:0'))

live_model.input_network.load_state_dict(torch.load(args.model_load_dir + f'{args.model_name}_input_network_{args.epoch}.pth', map_location='cuda:0'))
#live_model.eval()
#######
# dataset
txt_file_path = args.txt_file_path#'rl_model_train_data_for_torch_1011_cleaned_mini.txt'
dataset = LiveRecommendationDataset(txt_file_path, hash_size=hash_size, test_mode = True)
user_data_map  = {}
#import pdb
#pdb.set_trace()
def gen_batch_size_dim(data):
    for key,value in data.items():
        if key == "cur_features" or key == "nxt_features" or key == "cur_labels" or key == "nxt_labels":
            for kkey,vvalue in value.items():
                data[key][kkey] = [data[key][kkey]]
        else:
            data[key] = [data[key]]
for i in range(len(dataset)):
    data = dataset.__getitem__(i)
    #import pdb
    #pdb.set_trace()
    did = data['device_id']
    #data = pd.DataFrame.from_dict(data, orient='index').applymap(add_dimension).to_dict()
    gen_batch_size_dim(data)
    #print(data)
    #did = data['device_id']
    if did in user_data_map:
        user_data_map[did].append(data)
    else:
        user_data_map[did] = [data]
print("the number of total user trajactories: ",len(user_data_map))
# 实现后面的逻辑...
# import pdb
# pdb.set_trace()
#batch_size = 20

with torch.no_grad():
    R_list = []
    for did,trajactory_list in user_data_map.items():
        # print(did,trajactory_list)
        # break
        prob_traj=[]
        rewards=[]
        print(len(trajactory_list))
        for t in range(len(trajactory_list)):
            #print(trajactory_list[t])
            #break
            input = trajactory_list[t]
            cur_features, nxt_features, cur_labels, nxt_labels, not_final  = input['cur_features'],input['nxt_features'],input['cur_labels'],input['nxt_labels'], input['not_final']
        # print(not_final)
            #print(cur_features)
            state = input_network(cur_features)
            next_state = input_network(nxt_features)
            action = torch.tensor(cur_labels['reco_ban'],dtype=torch.int64).unsqueeze(1).cuda(args.device_name)
            reward = torch.tensor(cur_labels['time_delta']).unsqueeze(1).cuda(args.device_name)
            reward = F.sigmoid(0.1 * reward)
            done = torch.tensor(1-np.array(not_final)).unsqueeze(1).cuda(args.device_name)
            not_done = torch.tensor(not_final).unsqueeze(1).cuda(args.device_name)
            if alabtion:
                #import pdb
                #pdb.set_trace()
                #cur_action, cur_time_reward, cur_action_probs, actor_loss, time_rl_loss, critic_loss, sup_loss = live_model(input)
                cur_action_probs = live_model.get_probs(input)
            else:
                #import pdb
                #pdb.set_trace()
                if valueBased:
                    action_values, Q_loss = live_model(input)
                    cur_action_probs = F.softmax(action_values,dim=1) 
                else:
                    cur_action_probs, total_loss,Actor_loss = live_model(input)   
            if (action < 0).any() or (action >= cur_action_probs.size(1)).any():
                print(action)
            prob_traj.append(torch.gather(cur_action_probs,1,action).item())
            rewards.append(reward.item())
        #import pdb
        #pdb.set_trace()
        #prob_traj=np.array(prob_traj)
        norm_prob = 0
        if np.array(prob_traj).sum() != 0:
            norm_prob=np.array(prob_traj)/np.array(prob_traj).sum()
      
        R = (norm_prob*(np.array(rewards))).sum().item()
        if np.isnan(R):
            print(cur_action_probs)
            #break
        R_list.append(R)
        print("did:",did," ,R:",R)
        print(rewards)
    print("%s_model's sum of eval expectation rewards is: "%args.model_name,sum(R_list))
    with open('eval/%s_eval_expectation_rewards_per_user.txt'%args.model_name, 'w') as file:
        for item in R_list:
            file.write(f"{item}\n")
#     state,action,next_action,next_state,reward,done = trajactory
#     for array in [state,action,next_action,next_state,reward,done]:
#         assert not np.isnan(array).any()

#     for t in range(len(state)):
#         s=np.concatenate([state[t],h_state[t]])
#         policy_action=policy.select_action(s)                   
#         dist=Normal(policy_action, std)
        
#         a=torch.tensor(action[t]).to(device)
#         log_prob=dist.log_prob(a).mean(axis=-1) # should be sum, mean for larger value
#         prob=torch.exp(log_prob)
#         prob_traj.append(prob.cpu().item())
#         rewards.append(process_reward_eva(h_response[t],return_type))
#     prob_traj=np.array(prob_traj)
#     norm_prob=prob_traj/prob_traj.sum()
    
#     R=(norm_prob*(np.array(rewards))).sum().item()
        
#         # for t in range(len(norm_prob)):    
#         #     print(norm_prob[t])
            
#     with open('eval_expectation_rewards.txt', 'w') as file:
#        for item in R:
#            file.write(f"{item}\n")

