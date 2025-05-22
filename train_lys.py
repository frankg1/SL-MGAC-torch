import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch import nn
import torch
import torch.nn.functional as F
from random import randint
import numpy as np
from models.Base import InputNetwork
from torch.utils.tensorboard import SummaryWriter  # 引入 TensorBoard 的 SummaryWriter

from dataloaders.MyDataloader import LiveRecommendationDataset, collate_fn
################ import all the models
from models.MyModel import LiveRecoModel, step_counter
from models.MyModelK1 import LiveRecoModelK1
from models.MyModelK2 import LiveRecoModelK2
from models.MyModelK3 import LiveRecoModelK3
from models.MyModelK4 import LiveRecoModelK4
from models.MyModelK5 import LiveRecoModelK5

from models.Model_Gr import LiveRecoModelGr
from models.Model_GrBu import LiveRecoModelGrBu
from models.Model_GrBuSl import LiveRecoModelGrBuSl

from models.Model_LN import LiveRecoModelLN
from models.Model_SG import LiveRecoModelSG
from models.Model_SQ import LiveRecoModelSQ

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
parser.add_argument("--txt_file_path", type=str, default = 'rl_model_train_data_for_torch_1011_cleaned.txt', required = True)
parser.add_argument("--batch_size", type = int, default = 4, required = True)
parser.add_argument("--epoch_num", type = int, default = 50, required = True)
parser.add_argument("--enable_tb", type = bool, default = False, required = False)
parser.add_argument("--enable_save_model", type=bool, default=False, required=False)
parser.add_argument("--model_save_dir", type=str, default='save_weights/', required=False)
parser.add_argument("--model_name", type=str, default='sl_mgac',choices=["Model","ModelK1","ModelK2","ModelK3","ModelK4","ModelK5","Model_Gr","Model_GrBu","Model_GrBuSl","Model_LN","Model_SG","Model_SQ","BCQ","DQN","L2R","SAC","TD3","TD3_BC","IQL"], required=True)
parser.add_argument("--device_name", type=int, default=0, choices=[0,1], required=False)
args = parser.parse_args()
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
    "Model_GrBu": LiveRecoModelGrBu,
    "Model_GrBuSl": LiveRecoModelGrBuSl,
    "Model_LN": LiveRecoModelLN,
    "Model_SG": LiveRecoModelSG,
    "Model_SQ": LiveRecoModelSQ,
    "BCQ": discrete_BCQ,
    "DQN": DQN,
    "L2R": L2R2,
    "SAC": SAC,
    "TD3": TD3,
    "TD3_BC": TD3_BC,
    "IQL": IQL
}
alabtion = True if args.model_name in ["Model","ModelK1","ModelK2","ModelK3","ModelK4","ModelK5","Model_Gr","Model_GrBu","Model_GrBuSl","Model_LN","Model_SG","Model_SQ"] else False
valueBased = True if args.model_name in ["BCQ", "DQN", "L2R"] else False
########
# dataloader
txt_file_path = args.txt_file_path
dataset = LiveRecommendationDataset(txt_file_path)
data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
########
# input network定义，不要动！
embedding_dim=32
hash_size=5000
hidden_layers=[256,128]
input_network = InputNetwork(embedding_dim, hash_size, hidden_layers=hidden_layers,device_name=args.device_name)
input_network = input_network.cuda(args.device_name)
#########
# model定义 
live_model_class = model_mapping.get(args.model_name)
if live_model_class is None:
    raise ValueError(f"Unknown model name {args.model_name}")
if not alabtion:
    live_model = live_model_class(input_network=input_network, base_output_dim=128,device_name=args.device_name)
else:
    live_model = live_model_class(input_network=input_network, base_output_dim=128)
embedding_params = list(live_model.input_network.embedding_layer.parameters())  
other_params = [param for name, param in live_model.named_parameters() if "embedding" not in name]

optimizer = torch.optim.Adam([
    {'params': embedding_params, 'lr': 1e-5},
    {'params': other_params, 'lr': 1e-3}
])
##########
# tb
from datetime import datetime
now = datetime.now().strftime('%m_%d_%H_%M')
if args.enable_tb:
    writer = SummaryWriter(log_dir=f'runs/{args.model_name}_experiment{now}')
epoch_num = args.epoch_num
#########
#训练循环
if alabtion:
    for epoch in range(epoch_num):
        epoch_start_time = time()  # 记录 epoch 开始时间
        total_epoch_loss = 0  # 累计 epoch 的 loss
        print(f'Epoch: {epoch}')
        for i, batch in enumerate(data_loader):
            # 前向传播
            cur_action, cur_time_reward, cur_action_probs, actor_loss, time_rl_loss, critic_loss, sup_loss   = live_model(batch)
            total_loss = actor_loss + critic_loss
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            # 更新 step 计数器
            step_counter.increment_step()
            total_epoch_loss += total_loss.item()
            print(f'Batch {i} Loss: {total_loss.item()}, actor_loss: {actor_loss.item()}, time_rl_loss: {time_rl_loss.item()}, critic_loss: {critic_loss.item()}, sup_loss: {sup_loss.item()}')
            if args.enable_tb:
                writer.add_scalar('Loss/Total_Loss_per_Batch', total_loss.item(), step_counter.step)
                writer.add_scalar('Loss/Actor_Loss_per_Batch', actor_loss.item(), step_counter.step)
                writer.add_scalar('Loss/Time_RL_Loss_per_Batch', time_rl_loss.item(), step_counter.step)
                writer.add_scalar('Loss/Critic_Loss_per_Batch', critic_loss.item(), step_counter.step)
                writer.add_scalar('Loss/Sup_Loss_per_Batch', sup_loss.item(), step_counter.step)
        # 记录每个 epoch 的平均 loss
        avg_epoch_loss = total_epoch_loss / len(data_loader)
        print(f'Epoch {epoch} completed. Average Loss: {avg_epoch_loss:.4f}, Time: {time() - epoch_start_time:.2f}s')
        if args.enable_tb:
            writer.add_scalar('Loss/Average_Loss_per_Epoch', avg_epoch_loss, epoch)
            # 记录模型的参数和梯度分布到 TensorBoard
            for name, param in live_model.named_parameters():
                writer.add_histogram(f'Parameters/{name}', param, epoch)
                if param.grad is not None:
                    writer.add_histogram(f'Gradients/{name}', param.grad, epoch)

    
    if args.enable_save_model:
        model_save_dir = args.model_save_dir
        torch.save(live_model.input_network.state_dict(), model_save_dir + f'{args.model_name}_input_network.pth')
        torch.save(live_model.actor_network.state_dict(), model_save_dir + f'{args.model_name}_actor_network.pth')
        print(f"Model weights saved to {model_save_dir}")
else:
    # 非消融实验，返回值不同，没有actor网络，自行保存模型。
    for epoch in range(epoch_num):
        epoch_start_time = time()  # 记录 epoch 开始时间
        total_epoch_loss = 0  # 累计 epoch 的 loss
        print(f'Epoch: {epoch}')
        for i, batch in enumerate(data_loader):
            if valueBased:
                #import pdb
                #pdb.set_trace()
                _, Q_loss = live_model(batch)
                live_model.update_network(Q_loss,optimizer)
                step_counter.increment_step()
                total_epoch_loss += Q_loss.item()
                if args.enable_tb:
                    writer.add_scalar('Loss/Critic_Loss_per_Batch', Q_loss.item(), step_counter.step)
                print(f'Batch {i} Loss: critic_loss: {Q_loss.item()}')
            else:
                #with torch.autograd.detect_anomaly():
                _, Q_loss, Actor_loss = live_model(batch)
                total_loss = Q_loss + Actor_loss
                live_model.update_network(total_loss,optimizer)
                step_counter.increment_step()
                total_epoch_loss += total_loss.item()
                if args.enable_tb:
                    writer.add_scalar('Loss/Critic_Loss_per_Batch', Q_loss.item(), step_counter.step)
                    writer.add_scalar('Loss/Actor_Loss_per_Batch', Actor_loss.item(), step_counter.step)
                    writer.add_scalar('Loss/Total_Loss_per_Batch', total_loss.item(), step_counter.step)
                print(f'Batch {i} Loss: {total_loss.item()}, actor_loss: {Actor_loss.item()}, critic_loss: {Q_loss.item()}')
        # 记录每个 epoch 的平均 loss
        avg_epoch_loss = total_epoch_loss / len(data_loader)
        print(f'Epoch {epoch} completed. Average Loss: {avg_epoch_loss:.4f}, Time: {time() - epoch_start_time:.2f}s')
        if args.enable_tb:
            writer.add_scalar('Loss/Average_Loss_per_Epoch', avg_epoch_loss, epoch)
            # 记录模型的参数和梯度分布到 TensorBoard
            if args.model_name == "SAC": writer.add_scalar('SAC/alpha', live_model.alpha, epoch)          
            if args.model_name == "TD3_BC": writer.add_scalar('TD3_BC/bc_loss', live_model.bc_loss, epoch)
            if args.model_name == "IQL":
                 writer.add_scalar('IQL/V_loss', live_model.V_loss, epoch)
                 writer.add_scalar('IQL/critic_loss', live_model.critic_loss, epoch)
            for name, param in live_model.named_parameters():
                writer.add_histogram(f'Parameters/{name}', param, epoch)
                if param.grad is not None:
                     if torch.numel(param.grad) > 0:
                         try:
                             writer.add_histogram(f'Gradients/{name}', param.grad, epoch)
                         except:
                             print(param)
        
    if args.enable_save_model:
        model_save_dir = args.model_save_dir
        torch.save(live_model.input_network.state_dict(), model_save_dir + f'{args.model_name}_input_network.pth')
        try:
            torch.save(live_model.critic_network.state_dict(), model_save_dir + f'{args.model_name}_critic_network.pth')
        except:
            torch.save(live_model.critic_network1.state_dict(), model_save_dir + f'{args.model_name}_critic_network1.pth')
            torch.save(live_model.critic_network2.state_dict(), model_save_dir + f'{args.model_name}_critic_network2.pth')
        if not valueBased:
            torch.save(live_model.actor_network.state_dict(), model_save_dir + f'{args.model_name}_actor_network.pth')
        print(f"Model weights saved to {model_save_dir}")


if args.enable_tb:
    # 关闭 TensorBoard 记录器
    writer.close()


   
# 加载模型并测试推理
# 需要做适配和修改。
# print("Loading model for inference...")
# live_model_inference = LiveRecoModel()
# model_save_dir = args.model_save_dir
# # 分别加载权重
# live_model_inference.input_network.load_state_dict(torch.load(model_save_dir + f'{args.model_name}_input_network.pth', weights_only=True))
# live_model_inference.actor_network.load_state_dict(torch.load(model_save_dir + f'{args.model_name}_actor_network.pth', weights_only=True))
# live_model_inference.time_critic1.target_network = None
# live_model_inference.time_critic2.target_network = None
# live_model_inference.eval()  # 设置为评估模式


# # 从数据加载器中获取一个 batch 进行推理
# with torch.no_grad():
#     import torch.nn.functional as F
#     input = next(iter(data_loader))
#     cur_features, nxt_features, cur_labels, nxt_labels, not_final  = input['cur_features'],input['nxt_features'],input['cur_labels'],input['nxt_labels'], input['not_final']
#     cur_common_input_res = live_model_inference.input_network(cur_features)
#     user_type = cur_features['user_type'] # list: len = bs
#     user_type_indices = torch.tensor(user_type, dtype=torch.long).cuda(0) - 1       # [1, 6]--->范围是0~5
#     user_type_onehot = F.one_hot(user_type_indices, num_classes=6)  # [batch_size, user_type_num]
#     cur_action_logits = live_model_inference.actor_network(cur_common_input_res.detach(), user_type_onehot)
#     cur_action_probs = F.softmax(cur_action_logits, dim=1)
#     print(cur_action_probs)
