import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch import nn
import torch
import torch.nn.functional as F
from random import randint
import numpy as np

class DeviceController:
    def __init__(self):
        self.dvice_name = 0
    def set_device(self, i):
        self.dvice_name = i
    def get_device(self):
        if isinstance(self.dvice_name, int):
            return self.dvice_name
        else:
            return 0
device_controller = DeviceController()

class SimpleDenseNetwork(nn.Module):
    def __init__(self, input_dim, units, top_no_act=False, act = nn.ELU()):
        # top_no_act 控制top层是不是有激活函数
        super(SimpleDenseNetwork, self).__init__()
        layers = []
        in_dim = input_dim
        layer_num = len(units)
        for i, unit in enumerate(units):
            layers.append(nn.Linear(in_dim, unit))
            if not (top_no_act and i == layer_num - 1):
                layers.append(act)
            in_dim = unit
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
class InputNetwork(nn.Module):
    def __init__(self, embedding_dim, hash_size, hidden_layers):
        super(InputNetwork, self).__init__()
        self.embedding_layer = nn.Embedding(hash_size, embedding_dim)
        self.attn1 = nn.MultiheadAttention(embedding_dim * 4, num_heads=4)
        self.attn2 = nn.MultiheadAttention(embedding_dim * 5, num_heads=4)
        self.fc1 = nn.Linear(embedding_dim * 22, embedding_dim * 4)
        self.fc2 = nn.Linear(embedding_dim * 22, embedding_dim * 5)
        self.dense_layers = SimpleDenseNetwork(embedding_dim * 22 + (embedding_dim * 4) + (embedding_dim * 5), hidden_layers, top_no_act=True)
        
    def forward(self, features):
        batch_size = len(features['user_signs'])
        # features中包括 user_signs item_signs attn1_signs attn2_signs
        user_signs, item_signs, attn1_signs,attn2_signs = features["user_signs"],features["item_signs"],features["attn1_signs"],features["attn2_signs"]
        # List int --> embedding 
        user_embedding = self.embedding_layer(torch.LongTensor(user_signs).cuda(device_controller.get_device())).reshape((batch_size, -1))  #[bs, flatten]
        #import pdb
        #pdb.set_trace()
        item_embedding = self.embedding_layer(torch.LongTensor(item_signs).cuda(device_controller.get_device())).reshape((batch_size, -1))  #[bs, flatten]
        user_item_embedding = torch.cat((user_embedding ,item_embedding), dim = 1)
        # shape变换，仔细检查。。。。。
        attn1_embeddings = self.embedding_layer(torch.LongTensor(attn1_signs).cuda(device_controller.get_device())).view(batch_size, 4, 50, -1).permute(2, 0, 1, 3).reshape(50, batch_size, -1)
        attn2_embeddings = self.embedding_layer(torch.LongTensor(attn2_signs).cuda(device_controller.get_device())).view(batch_size, 5, 50, -1).permute(2, 0, 1, 3).reshape(50, batch_size, -1)
        fc1_out = self.fc1(user_item_embedding)  
        fc2_out = self.fc2(user_item_embedding)
        query1 = fc1_out.unsqueeze(0)
        query2 = fc2_out.unsqueeze(0)
        # print(query1.shape, attn1_embeddings.shape)
        # 基于 attn1_signs 和 attn2_signs 创建 key_padding_mask
        attn1_signs_tensor = torch.LongTensor(attn1_signs).view(batch_size, 4, 50).cuda(device_controller.get_device())  # [bs, 4, 50]
        attn2_signs_tensor = torch.LongTensor(attn2_signs).view(batch_size, 5, 50).cuda(device_controller.get_device())  # [bs, 5, 50]

        # 创建 key_padding_mask（维度: [batch_size, seq_length]）
        attn1_key_padding_mask = (attn1_signs_tensor == 0).all(dim=1)  # [bs, 50]
        attn2_key_padding_mask = (attn2_signs_tensor == 0).all(dim=1)
        
        # print(attn2_signs_tensor, attn2_key_padding_mask)
        
        attn1_output, attn1_weights = self.attn1(query1, attn1_embeddings, attn1_embeddings, key_padding_mask = attn1_key_padding_mask)
        attn2_output, attn2_weights = self.attn2(query2, attn2_embeddings, attn2_embeddings, key_padding_mask = attn2_key_padding_mask)
        
        attn1_output = attn1_output.squeeze(0)
        attn2_output = attn2_output.squeeze(0)
        fused_features = torch.cat((user_item_embedding, attn1_output, attn2_output), dim=1)
        final_output = self.dense_layers(fused_features)
        # print('out:',fc1_out,fc2_out,attn1_embeddings,attn2_embeddings,  fused_features,  final_output)
        return final_output

class MultiGroupNetwork(nn.Module):
    def __init__(self):
        super(MultiGroupNetwork, self).__init__()
        pass
    def forward(self, x):
        pass
