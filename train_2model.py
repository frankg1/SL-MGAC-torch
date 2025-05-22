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
from models.Base import device_controller

################ import all the models
from models.MyModel import LiveRecoModel, step_counter
from models.MyModelK1_saveq import LiveRecoModelK1
from models.Model_SGQ_saveq import LiveRecoModelSGQ
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
parser.add_argument("--model_name", type=str, default='sl_mgac',choices=["Model","ModelK1","ModelK2","ModelK3","ModelK4","ModelK5",
                                                                         "Model_Gr","Model_GrBu","Model_GrBuSl","Model_LN","Model_SG","Model_SQ",
                                                                         "Model_SGQ","Model_NotQ", "Model_SGSup", "Model_GrBuSlCopy","Model_GrLoss",
                                                                         "Model_GrCopy", "Model_GrBuCopy", "Model_SGQCopy"
                                                                         ], required=True)
parser.add_argument("--model_version", type = int, default = 0, required = True)
parser.add_argument("--device_name", type=int, default=0, required=False)

args = parser.parse_args()
device_controller.set_device(args.device_name)
##########
# mapping
# 新增加模型
# 1、需要在上面import
# 2、choices 增加model name
# 3、mapping中做好映射
model_mapping = {
    "ModelK1": LiveRecoModelK1,
    "Model_SGQ": LiveRecoModelSGQ,
}
alabtion = True if args.model_name in ["ModelK1","Model_SGQ"] else False

########
# dataloader
txt_file_path = args.txt_file_path
dataset = LiveRecommendationDataset(txt_file_path)
data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
########
# input network定义，不要动！
embedding_dim=32
hash_size=5000
hidden_layers=[256,128]
input_network = InputNetwork(embedding_dim, hash_size, hidden_layers=hidden_layers)
#########
# model定义 
live_model_class = model_mapping.get(args.model_name)
if live_model_class is None:
    raise ValueError(f"Unknown model name {args.model_name}")
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
    writer = SummaryWriter(log_dir=f'runs/{args.model_name}_experiment{now}_v{args.model_version}')
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
            cur_action, cur_time_reward, cur_action_probs, actor_loss, time_rl_loss, critic_loss, sup_loss,  cur_q1_action_value, cur_q2_action_value, q_label   = live_model(batch)
            total_loss = actor_loss + critic_loss
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            # 更新 step 计数器
            step_counter.increment_step()
            if  'Copy' in args.model_name and step_counter.get_step()%2 ==0:
                live_model._copy_critic_parameters()
            total_epoch_loss += total_loss.item()
            print(f'Batch {i} Loss: {total_loss.item()}, actor_loss: {actor_loss.item()}, time_rl_loss: {time_rl_loss.item()}, critic_loss: {critic_loss.item()}, sup_loss: {sup_loss.item()}')
            if args.enable_tb:
                writer.add_scalar('Loss/Total_Loss_per_Batch', total_loss.item(), step_counter.step)
                writer.add_scalar('Loss/Actor_Loss_per_Batch', actor_loss.item(), step_counter.step)
                writer.add_scalar('Loss/Time_RL_Loss_per_Batch', time_rl_loss.item(), step_counter.step)
                writer.add_scalar('Loss/Critic_Loss_per_Batch', critic_loss.item(), step_counter.step)
                writer.add_scalar('Loss/Sup_Loss_per_Batch', sup_loss.item(), step_counter.step)

                writer.add_scalar('Q_value/cur_q1_action_value', torch.sum(cur_q1_action_value).item(), step_counter.step)
                writer.add_scalar('Q_value/cur_q2_action_value', torch.sum(cur_q2_action_value).item(), step_counter.step)
                writer.add_scalar('Q_value/q_label', torch.sum(q_label).item(), step_counter.step)

                writer.add_scalar('Q_value_mean/cur_q1_action_value', torch.mean(cur_q1_action_value).item(),
                                  step_counter.step)
                writer.add_scalar('Q_value_mean/cur_q2_action_value', torch.mean(cur_q2_action_value).item(),
                                  step_counter.step)
                writer.add_scalar('Q_value_mean/q_label', torch.mean(q_label).item(), step_counter.step)

                writer.add_scalar('Q_value_var/cur_q1_action_value', torch.var(cur_q1_action_value).item(),
                                  step_counter.step)
                writer.add_scalar('Q_value_var/cur_q2_action_value', torch.var(cur_q2_action_value).item(),
                                  step_counter.step)
                writer.add_scalar('Q_value_var/q_label', torch.var(q_label).item(), step_counter.step)

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
        if args.enable_save_model and epoch % 50==0:
            model_save_dir = args.model_save_dir
            torch.save(live_model.input_network.state_dict(), model_save_dir + f'{args.model_name}_input_network_{epoch}.pth')
            torch.save(live_model.actor_network.state_dict(), model_save_dir + f'{args.model_name}_actor_network_{epoch}.pth')

            torch.save(live_model.time_critic1.critic1.state_dict(),
                       model_save_dir + f'{args.model_name}_time_critic1_network_1{epoch}.pth')
            torch.save(live_model.time_critic1.critic2.state_dict(),
                       model_save_dir + f'{args.model_name}_time_critic1_network_2{epoch}.pth')
            torch.save(live_model.time_critic2.critic1.state_dict(),
                       model_save_dir + f'{args.model_name}_time_critic2_network_1{epoch}.pth')
            torch.save(live_model.time_critic2.critic2.state_dict(),
                       model_save_dir + f'{args.model_name}_time_critic2_network_2{epoch}.pth')
            print(f"Epoch {epoch}: Model weights saved to {model_save_dir}")

    if args.enable_save_model:
        model_save_dir = args.model_save_dir
        torch.save(live_model.input_network.state_dict(), model_save_dir + f'{args.model_name}_input_network.pth')
        torch.save(live_model.actor_network.state_dict(), model_save_dir + f'{args.model_name}_actor_network.pth')
        print(f"Model weights saved to {model_save_dir}")
else:
    print('没有else，检查代码是不是写错了！')

if args.enable_tb:
    # 关闭 TensorBoard 记录器
    writer.close()