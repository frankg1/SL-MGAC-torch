import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch import nn
import torch
import torch.nn.functional as F
from random import randint
import numpy as np
class LiveRecommendationDataset(Dataset):
    def __init__(self, data_file, hash_size=5000, test_mode = False): 
        self.test_mode = test_mode
        self.data = pd.read_csv(data_file, sep='\x01', header=None, na_values='\\N').fillna(0)
        if self.test_mode:
            self.data = self.data.sort_values(by=[self.data.columns[1], self.data.columns[3]])
        self.features = {
            'cur_features': {
                'common_signs': list(self.data.iloc[:, 71]),  # 71
                'item_signs': list(self.data.iloc[:, 75]),   # 75
                'common_slots': list(self.data.iloc[:, 70]),  # 70
                'item_slots': list(self.data.iloc[:, 74]),   # 74
                'user_type': list(self.data.iloc[:, 15])      # 15
            },
            'nxt_features': {
                'common_signs': list(self.data.iloc[:, 73]),  # 73
                'item_signs': list(self.data.iloc[:, 77]),   # 77
                'common_slots': list(self.data.iloc[:, 72]),  # 72
                'item_slots': list(self.data.iloc[:, 76]),   # 76
                'user_type': list(self.data.iloc[:, 35])      # 25
            }
        }

        self.labels = {
            "cur_labels":{
                'live_end_auto_watch': list(self.data.iloc[:, 45]), # 45
                'live_end_watch': list(self.data.iloc[:, 47]), # 47
                'video_play_time': list(self.data.iloc[:, 17]), # 17
                'video_play_cnt': list(self.data.iloc[:, 18]), # 18
                'reco_ban': list(self.data.iloc[:, 66]), # 66
                ### 真正的label信息定义如下：
                'time_ratio': [],
                'time_reward': [],
                'time_idx_vec': [],
                'time_reward_start': [],
                'time_reward_range': [],
                'time_delta': [],
                
                'photo_ratio': [],
                'photo_reward': [],
                'photo_idx_vec': [],
                'photo_reward_start': [],
                'photo_reward_range': []
            },
            "nxt_labels":{
                'live_end_auto_watch': list(self.data.iloc[:, 84]), # 84
                'live_end_watch': list(self.data.iloc[:, 86]), # 86
                'video_play_time': list(self.data.iloc[:, 90]), # 90
                'video_play_cnt': list(self.data.iloc[:, 91]), # 91
                #'reco_ban': [randint(0, 1) for _ in range(self.data.shape[0])],#暂时没有, mock吧
                'reco_ban':list(self.data.iloc[:, 93]),
                ### 真正的label信息定义如下：
                'time_ratio': [],
                'time_reward': [],
                'time_idx_vec': [],
                'time_reward_start': [],
                'time_reward_range': [],
                'time_delta': [],
                
                'photo_ratio': [],
                'photo_reward': [],
                'photo_idx_vec': [],
                'photo_reward_start': [],
                'photo_reward_range': []
            },
            "not_final": list(self.data.iloc[:, 16]) #18
        }
        if self.test_mode:
            self.user_info = {
                "request_time": list(self.data.iloc[:, 3]),
                "device_id": list(self.data.iloc[:, 1])
                }
        self.hash_size = hash_size

        self.user_type_mapping = {
            'Core': 1,
            'Gift': 2,
            'Potential_old': 3,
            'Potential_new': 4,
            'Long_old': 5,
            'Long_new': 6
        }
        #调用_process_labels
        self._process_labels()

    def __len__(self):
        return len(self.features['cur_features']['common_signs'])
    
    def _get_features(self, 
                        common_sign_list_str, 
                        common_slot_list_str, 
                        item_sign_list_str, 
                        item_slot_list_str,
                        user_slots,
                        item_slots,
                        attn1_slots,
                        attn2_slots):
        common_sign_list = self._parse_signs(common_sign_list_str)
        common_slot_list = self._parse_slots(common_slot_list_str)
        
        item_sign_list = self._parse_signs(item_sign_list_str)
        item_slot_list = self._parse_slots(item_slot_list_str)
        
        common_slot_dict = {}
        for slot, sign in zip(common_slot_list, common_sign_list):
            if slot in common_slot_dict:
                common_slot_dict[slot].append(sign)
            else:
                common_slot_dict[slot] = [sign]
        # key -value  slot:[signs]
        item_slot_dict = {slot: sign for slot, sign in zip(item_slot_list, item_sign_list)}
        
        cur_user_signs, cur_item_signs, cur_attn1_signs, cur_attn2_signs = [], [], [], []
        
        # 处理 cur_user_signs
        for slot in user_slots:
            signs = common_slot_dict.get(slot, [0])
            cur_user_signs.append(signs[0] if signs else 0)
        
        # 处理 cur_item_signs
        for slot in item_slots:
            cur_item_signs.append(item_slot_dict.get(slot, 0))
        
        def generate_attn_signs_per_slot(slots, slot_dict, max_length=50):
            attn_signs = []
            for slot in slots:
                signs = slot_dict.get(slot, [])
                # 如果slot数量不足50，填充0；如果超过50，截断
                if len(signs) >= max_length:
                    padded_signs = signs[:max_length]
                else:
                    padded_signs = signs + [0] * (max_length - len(signs))
                attn_signs.extend(padded_signs)
            return attn_signs
        
        # 处理 cur_attn1_signs 和 cur_attn2_signs
        cur_attn1_signs = generate_attn_signs_per_slot(attn1_slots, common_slot_dict, 50)
        cur_attn2_signs = generate_attn_signs_per_slot(attn2_slots, common_slot_dict, 50)
        
        return cur_user_signs, cur_item_signs, cur_attn1_signs, cur_attn2_signs
    def _process_labels(self):
        '''
        return [cur labels]:time_radio,time_reward,idx_vec,start,range,photo,,,,[nxt labels]10个
        '''
        w_live_time_ban = 0.34
        def calc_reward_info(time, reco_ban, ban_reward):
            ratio = 0.0
            reward = ban_reward
            max_thres = 1.0
            time_list = [0.0, 6.0, 15.0, 30.0, 60.0, 100.0, 600.0, 1200.0]
            reward_list = [0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.9, 1.0]
            bucket_num = len(time_list) - 1
            idx_vec = [0, bucket_num]
            reward_start = [0.0, 0.0]
            reward_range = [6.0, max_thres]
            if reco_ban == 1:
                for i in range(bucket_num):
                    if reward >= reward_list[i] and reward < reward_list[i + 1]:
                        reward_ratio = (reward - reward_list[i]) / (reward_list[i + 1] - reward_list[i])
                        ban_time = time_list[i] + reward_ratio * (time_list[i + 1] - time_list[i])
                        ratio = ban_time / max_thres
                        break
                #ban的情况下：  ratio=0, idx_vec=[0,8], reward_start = [0.0, 0.0] reward_range=[6.0, 1.0] reward = 0.0
                return ratio, idx_vec, reward_start, reward_range, reward
            #不ban
            for i in range(bucket_num):
                # i最大是6
                if time >= time_list[i] and time < time_list[i + 1]:
                    idx_vec = [i, bucket_num]
                    reward_start = [time_list[i], 0.0]
                    reward_range = [time_list[i + 1] - time_list[i], max_thres]
                    ratio = (time - time_list[i]) / (time_list[i + 1] - time_list[i])
                    reward = reward_list[i] + ratio * (reward_list[i + 1] - reward_list[i])
                    break
            # 假设 time=10s i=1 ,落在 1 2之间（0开始）
            # idx_vec = [1, 8] reward_start=[6.0, 0.0], reward_range = [15-6, 1.0], ratio = (10 -6) / (15 - 6); reward = 0.2 * ratio*(0.4 - 0.2)
            return ratio, idx_vec, reward_start, reward_range, reward
            
        def gen_time_reward(self):
            live_duration = (np.array(self.labels['cur_labels']['live_end_auto_watch']) + np.array(self.labels['cur_labels']['live_end_watch'])).tolist()
            reco_ban = self.labels['cur_labels']['reco_ban']
            nxt_live_duration = (np.array(self.labels['nxt_labels']['live_end_auto_watch']) + np.array(self.labels['nxt_labels']['live_end_watch'])).tolist()
            nxt_reco_ban = self.labels['nxt_labels']['reco_ban']
            
            ratio, time_idx_vec, time_reward_start, time_reward_range, time_reward, nxt_ratio, nxt_time_idx_vec, nxt_time_reward_start, nxt_time_reward_range, nxt_time_reward =[[0 for _ in range(len(reco_ban))] for _ in range(10)]
            # print('init: ', ratio,time_idx_vec)
            for i in range(len(reco_ban)):
                ratio[i], time_idx_vec[i], time_reward_start[i], time_reward_range[i], time_reward[i] = calc_reward_info(live_duration[i], reco_ban[i], 0.0)
                nxt_ratio[i], nxt_time_idx_vec[i], nxt_time_reward_start[i], nxt_time_reward_range[i], nxt_time_reward[i] = calc_reward_info(nxt_live_duration[i], nxt_reco_ban[i], 0.0)
            # 填充
            self.labels['cur_labels']['time_ratio'] = ratio
            self.labels['cur_labels']['time_reward'] = time_reward
            self.labels['cur_labels']['time_idx_vec'] = time_idx_vec
            self.labels['cur_labels']['time_reward_start'] = time_reward_start
            self.labels['cur_labels']['time_reward_range'] = time_reward_range
            
            self.labels['nxt_labels']['time_ratio'] = nxt_ratio
            self.labels['nxt_labels']['time_reward'] = nxt_time_reward
            self.labels['nxt_labels']['time_idx_vec'] = nxt_time_idx_vec
            self.labels['nxt_labels']['time_reward_start'] = nxt_time_reward_start
            self.labels['nxt_labels']['time_reward_range'] = nxt_time_reward_range
        
        def calc_photo_reward_info(time, reco_ban, ban_reward):       
            ratio = 0.0
            reward = ban_reward
            ban_time = 0.0
            max_thres = 25.0
            time_list = [0.0, 3.0, 10.0, 25.0, 50.0, 100.0, 600.0, 1200.0]
            reward_list = [0.0, 0.1, 0.3, 0.5, 0.6, 0.7, 0.9, 1.0]
            bucket_num = len(time_list) -1
            idx_vec = [2, bucket_num]
            reward_start = [10.0, 0.0]
            reward_range = [15.0, max_thres]
            
            if reco_ban == 1:
                for i in range(bucket_num):
                    if reward >= reward_list[i] and reward < reward_list[i + 1]:
                        reward_ratio = (reward - reward_list[i]) / (reward_list[i + 1] - reward_list[i])
                        ban_time = time_list[i] + reward_ratio * (time_list[i + 1] - time_list[i])
                        ratio = ban_time / max_thres
                        break
                return ratio, ban_time, idx_vec, reward_start, reward_range, reward
        
            for i in range(bucket_num):
                if time >= time_list[i] and time < time_list[i + 1]:
                    idx_vec = [i, bucket_num]
                    reward_start = [time_list[i], 0.0]
                    reward_range = [time_list[i + 1] - time_list[i], max_thres]
                    ratio = (time - time_list[i]) / (time_list[i + 1] - time_list[i])
                    reward = reward_list[i] + ratio * (reward_list[i + 1] - reward_list[i])
                    break
            return ratio, ban_time, idx_vec, reward_start, reward_range, reward
        
        def gen_photo_reward(self):
            reco_ban = self.labels['cur_labels']['reco_ban']
            nxt_reco_ban = self.labels['nxt_labels']['reco_ban']
            avg_photo_play_time = (np.array(self.labels['cur_labels']['video_play_time']) / (np.array(self.labels['cur_labels']['video_play_cnt']) + 1e-8)).tolist()
            
            nxt_avg_photo_play_time = (np.array(self.labels['nxt_labels']['video_play_time'])  / (np.array(self.labels['nxt_labels']['video_play_cnt']) + 1e-8)).tolist()
            time_reco_ban_reward = w_live_time_ban 
            
            ratio, photo_ban_time, photo_idx_vec, photo_reward_start, photo_reward_range, photo_reward, nxt_ratio, nxt_photo_ban_time, nxt_photo_idx_vec, nxt_photo_reward_start, nxt_photo_reward_range, nxt_photo_reward = [[0 for _ in range(len(reco_ban))] for _ in range(12)]
            for i in range(len(reco_ban)):
                ratio[i], photo_ban_time[i], photo_idx_vec[i], photo_reward_start[i], photo_reward_range[i], photo_reward[i] = calc_photo_reward_info(avg_photo_play_time[i], reco_ban[i], time_reco_ban_reward)
            
                nxt_ratio[i], nxt_photo_ban_time[i], nxt_photo_idx_vec[i], nxt_photo_reward_start[i], nxt_photo_reward_range[i], nxt_photo_reward[i] = calc_photo_reward_info(nxt_avg_photo_play_time[i], nxt_reco_ban[i], time_reco_ban_reward)
            # 填充
            self.labels['cur_labels']['photo_ratio'] = ratio
            self.labels['cur_labels']['photo_reward'] = photo_reward
            self.labels['cur_labels']['photo_idx_vec'] = photo_idx_vec
            self.labels['cur_labels']['photo_reward_start'] = photo_reward_start
            self.labels['cur_labels']['photo_reward_range'] = photo_reward_range
            
            self.labels['nxt_labels']['photo_ratio'] = nxt_ratio
            self.labels['nxt_labels']['photo_reward'] = nxt_photo_reward
            self.labels['nxt_labels']['photo_idx_vec'] = nxt_photo_idx_vec
            self.labels['nxt_labels']['photo_reward_start'] = nxt_photo_reward_start
            self.labels['nxt_labels']['photo_reward_range'] = nxt_photo_reward_range
            return photo_ban_time, nxt_photo_ban_time
        def gen_time_delta(self):
            photo_ban_time, nxt_photo_ban_time = gen_photo_reward(self)
            
            live_duration = (np.array(self.labels['cur_labels']['live_end_auto_watch'])  + np.array(self.labels['cur_labels']['live_end_watch'])).tolist() 
            nxt_live_duration = (np.array(self.labels['nxt_labels']['live_end_auto_watch']) + np.array(self.labels['nxt_labels']['live_end_watch'])).tolist()
            avg_photo_play_time = (np.array(self.labels['cur_labels']['video_play_time']) / (np.array(self.labels['cur_labels']['video_play_cnt']) + 1e-8)).tolist()
            nxt_avg_photo_play_time = (np.array(self.labels['nxt_labels']['video_play_time']) / (np.array(self.labels['nxt_labels']['video_play_cnt']) + 1e-8)).tolist()
            reco_ban = self.labels['cur_labels']['reco_ban']
            nxt_reco_ban = self.labels['nxt_labels']['reco_ban']
            time_diff = (np.array(live_duration) - np.array(avg_photo_play_time)).tolist()
            nxt_time_diff = (np.array(nxt_live_duration) - np.array(nxt_avg_photo_play_time)).tolist()
            for idx in range(len(reco_ban)):
                if reco_ban[idx] ==1:
                    time_diff[idx] = -1 * photo_ban_time[idx]
                if nxt_reco_ban[idx] == 1:
                    nxt_time_diff[idx] = -1 * nxt_photo_ban_time[idx]  
            self.labels['cur_labels']['time_delta'] = time_diff
            self.labels['nxt_labels']['time_delta'] = nxt_time_diff
        
        def process_second(self):
            self.labels['cur_labels']['live_end_auto_watch']  = (np.array(self.labels['cur_labels']['live_end_auto_watch']) / 1000).tolist()
            self.labels['cur_labels']['live_end_watch']  = (np.array(self.labels['cur_labels']['live_end_watch']) / 1000).tolist()
            self.labels['cur_labels']['video_play_time']  = (np.array(self.labels['cur_labels']['video_play_time']) / 1000).tolist()
            
            self.labels['nxt_labels']['live_end_auto_watch']  = (np.array(self.labels['nxt_labels']['live_end_auto_watch']) / 1000).tolist()
            self.labels['nxt_labels']['live_end_watch']  = (np.array(self.labels['nxt_labels']['live_end_watch']) / 1000).tolist()
            self.labels['nxt_labels']['video_play_time']  = (np.array(self.labels['nxt_labels']['video_play_time']) / 1000).tolist()
            
        process_second(self)
        gen_time_reward(self)
        gen_time_delta(self) #其中调用了gen_photo_reward
    
    def __getitem__(self, idx):
        # self.features['cur_features']['common_signs'][idx], self.features['cur_features']['common_slots'][idx], self.features['cur_features']['item_signs'][idx], self.features['cur_features']['item_slots'][idx],
        cur_user_slots =  [1000, 1001, 1008, 1009, 1020, 1004, 1005, 1006, 1007, 1200, 1201, 1202]
        cur_item_slots =  [100, 101, 102, 103, 105, 106, 107, 108, 109, 110]
        cur_attn1_slots = [1145, 1101, 1102, 1147]
        cur_attn2_slots = [1149, 1103, 1104, 1105, 1106]
        
        nxt_user_slots =  [1010, 1011, 1018, 1019, 1021, 1014, 1015, 1016, 1017, 1203, 1204, 1205]
        #nxt_item_slots =  [112, 113, 114, 115, 117, 118, 119, 120, 121, 122]
        nxt_item_slots = [100, 101, 102, 103, 105, 106, 107, 108, 109, 110]   #修改成cur的，因为数据源没有做映射。
        nxt_attn1_slots = [1146, 1123, 1124, 1148]
        nxt_attn2_slots = [1150, 1125, 1126, 1127, 1128]
        
        cur_user_signs, cur_item_signs, cur_attn1_signs, cur_attn2_signs = self._get_features(
            self.features['cur_features']['common_signs'][idx], 
            self.features['cur_features']['common_slots'][idx], 
            self.features['cur_features']['item_signs'][idx], 
            self.features['cur_features']['item_slots'][idx],
            cur_user_slots,
            cur_item_slots,
            cur_attn1_slots,
            cur_attn2_slots
        )
        nxt_user_signs, nxt_item_signs, nxt_attn1_signs, nxt_attn2_signs = self._get_features(
            self.features['nxt_features']['common_signs'][idx], 
            self.features['nxt_features']['common_slots'][idx], 
            self.features['nxt_features']['item_signs'][idx], 
            self.features['nxt_features']['item_slots'][idx],
            nxt_user_slots,
            nxt_item_slots,
            nxt_attn1_slots,
            nxt_attn2_slots
        )
        if not self.test_mode:
            sample = {
                'cur_features': {
                    'user_signs': cur_user_signs,
                    'item_signs': cur_item_signs,
                    'attn1_signs': cur_attn1_signs,
                    'attn2_signs': cur_attn2_signs,
                    'user_type': self._map_user_type(self.features['cur_features']['user_type'][idx])
                },
                'nxt_features': {
                    'user_signs': nxt_user_signs,
                    'item_signs': nxt_item_signs,
                    'attn1_signs':nxt_attn1_signs,
                    'attn2_signs':nxt_attn2_signs,
                    'user_type': self._map_user_type(self.features['nxt_features']['user_type'][idx])
                },
                'cur_labels': {
                    'live_end_auto_watch': self.labels['cur_labels']['live_end_auto_watch'][idx],
                    'live_end_watch': self.labels['cur_labels']['live_end_watch'][idx],
                    'video_play_time': self.labels['cur_labels']['video_play_time'][idx],  # 填充 video_play_time
                    'video_play_cnt': self.labels['cur_labels']['video_play_cnt'][idx],  # 填充 video_play_cnt
                    'reco_ban': self.labels['cur_labels']['reco_ban'][idx],  # 添加 reco_ban 标签
                    'time_ratio': self.labels['cur_labels']['time_ratio'][idx] if len(self.labels['cur_labels']['time_ratio']) > 0 else None,
                    'time_reward': self.labels['cur_labels']['time_reward'][idx] if len(self.labels['cur_labels']['time_reward']) > 0 else None,
                    'time_idx_vec': self.labels['cur_labels']['time_idx_vec'][idx] if len(self.labels['cur_labels']['time_idx_vec']) > 0 else None,
                    'time_reward_start': self.labels['cur_labels']['time_reward_start'][idx] if len(self.labels['cur_labels']['time_reward_start']) > 0 else None,
                    'time_reward_range': self.labels['cur_labels']['time_reward_range'][idx] if len(self.labels['cur_labels']['time_reward_range']) > 0 else None,
                    'time_delta': self.labels['cur_labels']['time_delta'][idx] if len(self.labels['cur_labels']['time_delta']) > 0 else None,
                    'photo_ratio': self.labels['cur_labels']['photo_ratio'][idx] if len(self.labels['cur_labels']['photo_ratio']) > 0 else None,
                    'photo_reward': self.labels['cur_labels']['photo_reward'][idx] if len(self.labels['cur_labels']['photo_reward']) > 0 else None,
                    'photo_idx_vec': self.labels['cur_labels']['photo_idx_vec'][idx] if len(self.labels['cur_labels']['photo_idx_vec']) > 0 else None,
                    'photo_reward_start': self.labels['cur_labels']['photo_reward_start'][idx] if len(self.labels['cur_labels']['photo_reward_start']) > 0 else None,
                    'photo_reward_range': self.labels['cur_labels']['photo_reward_range'][idx] if len(self.labels['cur_labels']['photo_reward_range']) > 0 else None
                },
                'nxt_labels': {
                    'live_end_auto_watch': self.labels['nxt_labels']['live_end_auto_watch'][idx],
                    'live_end_watch': self.labels['nxt_labels']['live_end_watch'][idx],
                    'video_play_time': self.labels['nxt_labels']['video_play_time'][idx],  # 填充 video_play_time
                    'video_play_cnt': self.labels['nxt_labels']['video_play_cnt'][idx],  # 填充 video_play_cnt
                    'reco_ban': self.labels['nxt_labels']['reco_ban'][idx],  # 添加 reco_ban 标签
                    'time_ratio': self.labels['nxt_labels']['time_ratio'][idx] if len(self.labels['nxt_labels']['time_ratio']) > 0 else None,
                    'time_reward': self.labels['nxt_labels']['time_reward'][idx] if len(self.labels['nxt_labels']['time_reward']) > 0 else None,
                    'time_idx_vec': self.labels['nxt_labels']['time_idx_vec'][idx] if len(self.labels['nxt_labels']['time_idx_vec']) > 0 else None,
                    'time_reward_start': self.labels['nxt_labels']['time_reward_start'][idx] if len(self.labels['nxt_labels']['time_reward_start']) > 0 else None,
                    'time_reward_range': self.labels['nxt_labels']['time_reward_range'][idx] if len(self.labels['nxt_labels']['time_reward_range']) > 0 else None,
                    'time_delta': self.labels['nxt_labels']['time_delta'][idx] if len(self.labels['nxt_labels']['time_delta']) > 0 else None,
                    'photo_ratio': self.labels['nxt_labels']['photo_ratio'][idx] if len(self.labels['nxt_labels']['photo_ratio']) > 0 else None,
                    'photo_reward': self.labels['nxt_labels']['photo_reward'][idx] if len(self.labels['nxt_labels']['photo_reward']) > 0 else None,
                    'photo_idx_vec': self.labels['nxt_labels']['photo_idx_vec'][idx] if len(self.labels['nxt_labels']['photo_idx_vec']) > 0 else None,
                    'photo_reward_start': self.labels['nxt_labels']['photo_reward_start'][idx] if len(self.labels['nxt_labels']['photo_reward_start']) > 0 else None,
                    'photo_reward_range': self.labels['nxt_labels']['photo_reward_range'][idx] if len(self.labels['nxt_labels']['photo_reward_range']) > 0 else None
                },
                'not_final': self.labels['not_final'][idx]
            }
        else:
            sample = {
                'cur_features': {
                    'user_signs': cur_user_signs,
                    'item_signs': cur_item_signs,
                    'attn1_signs': cur_attn1_signs,
                    'attn2_signs': cur_attn2_signs,
                    'user_type': self._map_user_type(self.features['cur_features']['user_type'][idx])
                },
                'nxt_features': {
                    'user_signs': nxt_user_signs,
                    'item_signs': nxt_item_signs,
                    'attn1_signs':nxt_attn1_signs,
                    'attn2_signs':nxt_attn2_signs,
                    'user_type': self._map_user_type(self.features['nxt_features']['user_type'][idx])
                },
                'cur_labels': {
                    'live_end_auto_watch': self.labels['cur_labels']['live_end_auto_watch'][idx],
                    'live_end_watch': self.labels['cur_labels']['live_end_watch'][idx],
                    'video_play_time': self.labels['cur_labels']['video_play_time'][idx],  # 填充 video_play_time
                    'video_play_cnt': self.labels['cur_labels']['video_play_cnt'][idx],  # 填充 video_play_cnt
                    'reco_ban': self.labels['cur_labels']['reco_ban'][idx],  # 添加 reco_ban 标签
                    'time_ratio': self.labels['cur_labels']['time_ratio'][idx] if len(self.labels['cur_labels']['time_ratio']) > 0 else None,
                    'time_reward': self.labels['cur_labels']['time_reward'][idx] if len(self.labels['cur_labels']['time_reward']) > 0 else None,
                    'time_idx_vec': self.labels['cur_labels']['time_idx_vec'][idx] if len(self.labels['cur_labels']['time_idx_vec']) > 0 else None,
                    'time_reward_start': self.labels['cur_labels']['time_reward_start'][idx] if len(self.labels['cur_labels']['time_reward_start']) > 0 else None,
                    'time_reward_range': self.labels['cur_labels']['time_reward_range'][idx] if len(self.labels['cur_labels']['time_reward_range']) > 0 else None,
                    'time_delta': self.labels['cur_labels']['time_delta'][idx] if len(self.labels['cur_labels']['time_delta']) > 0 else None,
                    'photo_ratio': self.labels['cur_labels']['photo_ratio'][idx] if len(self.labels['cur_labels']['photo_ratio']) > 0 else None,
                    'photo_reward': self.labels['cur_labels']['photo_reward'][idx] if len(self.labels['cur_labels']['photo_reward']) > 0 else None,
                    'photo_idx_vec': self.labels['cur_labels']['photo_idx_vec'][idx] if len(self.labels['cur_labels']['photo_idx_vec']) > 0 else None,
                    'photo_reward_start': self.labels['cur_labels']['photo_reward_start'][idx] if len(self.labels['cur_labels']['photo_reward_start']) > 0 else None,
                    'photo_reward_range': self.labels['cur_labels']['photo_reward_range'][idx] if len(self.labels['cur_labels']['photo_reward_range']) > 0 else None
                },
                'nxt_labels': {
                    'live_end_auto_watch': self.labels['nxt_labels']['live_end_auto_watch'][idx],
                    'live_end_watch': self.labels['nxt_labels']['live_end_watch'][idx],
                    'video_play_time': self.labels['nxt_labels']['video_play_time'][idx],  # 填充 video_play_time
                    'video_play_cnt': self.labels['nxt_labels']['video_play_cnt'][idx],  # 填充 video_play_cnt
                    'reco_ban': self.labels['nxt_labels']['reco_ban'][idx],  # 添加 reco_ban 标签
                    'time_ratio': self.labels['nxt_labels']['time_ratio'][idx] if len(self.labels['nxt_labels']['time_ratio']) > 0 else None,
                    'time_reward': self.labels['nxt_labels']['time_reward'][idx] if len(self.labels['nxt_labels']['time_reward']) > 0 else None,
                    'time_idx_vec': self.labels['nxt_labels']['time_idx_vec'][idx] if len(self.labels['nxt_labels']['time_idx_vec']) > 0 else None,
                    'time_reward_start': self.labels['nxt_labels']['time_reward_start'][idx] if len(self.labels['nxt_labels']['time_reward_start']) > 0 else None,
                    'time_reward_range': self.labels['nxt_labels']['time_reward_range'][idx] if len(self.labels['nxt_labels']['time_reward_range']) > 0 else None,
                    'time_delta': self.labels['nxt_labels']['time_delta'][idx] if len(self.labels['nxt_labels']['time_delta']) > 0 else None,
                    'photo_ratio': self.labels['nxt_labels']['photo_ratio'][idx] if len(self.labels['nxt_labels']['photo_ratio']) > 0 else None,
                    'photo_reward': self.labels['nxt_labels']['photo_reward'][idx] if len(self.labels['nxt_labels']['photo_reward']) > 0 else None,
                    'photo_idx_vec': self.labels['nxt_labels']['photo_idx_vec'][idx] if len(self.labels['nxt_labels']['photo_idx_vec']) > 0 else None,
                    'photo_reward_start': self.labels['nxt_labels']['photo_reward_start'][idx] if len(self.labels['nxt_labels']['photo_reward_start']) > 0 else None,
                    'photo_reward_range': self.labels['nxt_labels']['photo_reward_range'][idx] if len(self.labels['nxt_labels']['photo_reward_range']) > 0 else None
                },
                'not_final': self.labels['not_final'][idx],
                'request_time': self.user_info['request_time'][idx],
                'device_id':self.user_info['device_id'][idx]
            }
        return sample

    def _parse_signs(self, sign_str):
        '''
            处理sign，并且给他hash到一个范围内。
            根据我的统计：
                一共1000w sign。
                出现5次以上的100w.
                出现10次以上65w。
            结论：sign的分布非常的不均衡，所以直接给他hash映射就行了，小频率的sign不影响embedding layer的训练。
        '''
        # print(sign_str, type(sign_str))
        if not isinstance(sign_str, str):
            return []
        if pd.isna(sign_str) or sign_str.strip() == "[]":
            return []
        sign_str = sign_str.strip('[]')
        sign_list = sign_str.split(',')
        signs = [self._mod_sign(int(sign.strip())) for sign in sign_list if sign.strip().isdigit()]
        return signs

    def _parse_slots(self, slot_str):
        #处理slot，返回list
        if not isinstance(slot_str, str):
            return []
        if pd.isna(slot_str) or slot_str.strip() == "[]":
            return []
        slot_str = slot_str.strip('[]')
        slot_list = slot_str.split(',')
        return [int(slot.strip()) for slot in slot_list if slot.strip().isdigit()]
    def _map_user_type(self, user_type_str):
        try:
            res = self.user_type_mapping.get(user_type_str.strip(), 0)
        except Exception as e:
            # print(user_type_str, type(user_type_str))
            res = 6
        return res
    def _mod_sign(self, sign):
        return (sign % (self.hash_size - 1)) + 1
    def _check_data(self):
        for row_num in range(self.data.shape[0]):
            sign_str = self.data.iloc[row_num, 71]
            sign_str1 = self.data.iloc[row_num, 15]
            sign_str2 = self.data.iloc[row_num, 35]
            if not isinstance(sign_str, str):
                self.data.drop(row_num, inplace=True)
            if isinstance(sign_str1, int):
                self.data.drop(row_num, inplace=True)
            if isinstance(sign_str2, int):
                self.data.drop(row_num, inplace=True)

def collate_fn(batch):
    # 初始化各个batch的特征和标签
    batched_cur_features = {
        'user_signs': [],
        'item_signs': [],
        'attn1_signs': [],
        'attn2_signs': [],
        'user_type': [],
    }
    batched_nxt_features = {
        'user_signs': [],
        'item_signs': [],
        'attn1_signs': [],
        'attn2_signs': [],
        'user_type': [],
    }
    
    batched_cur_labels = {
        'live_end_auto_watch': [],
        'live_end_watch': [],
        'video_play_time': [],
        'video_play_cnt': [],
        'reco_ban': [],
        'time_ratio': [],
        'time_reward': [],
        'time_idx_vec': [],
        'time_reward_start': [],
        'time_reward_range': [],
        'time_delta':[],
        'photo_ratio': [],
        'photo_reward': [],
        'photo_idx_vec': [],
        'photo_reward_start': [],
        'photo_reward_range': []
    }
    batched_nxt_labels = {
        'live_end_auto_watch': [],
        'live_end_watch': [],
        'video_play_time': [],
        'video_play_cnt': [],
        'reco_ban': [],
        'time_ratio': [],
        'time_reward': [],
        'time_idx_vec': [],
        'time_reward_start': [],
        'time_reward_range': [],
        'time_delta':[],
        'photo_ratio': [],
        'photo_reward': [],
        'photo_idx_vec': [],
        'photo_reward_start': [],
        'photo_reward_range': []
    }
    not_final = []

    # 开始填充batch
    for sample in batch:
        # 填充 cur_features 和 nxt_features
        for key in batched_cur_features.keys():
            batched_cur_features[key].append(sample['cur_features'][key])
            batched_nxt_features[key].append(sample['nxt_features'][key])

        # 填充 cur_labels 和 nxt_labels
        for key in batched_cur_labels.keys():
            batched_cur_labels[key].append(sample['cur_labels'][key])
            batched_nxt_labels[key].append(sample['nxt_labels'][key])

        # 填充 not_final 标签
        not_final.append(sample['not_final'])

    return {
        'cur_features': batched_cur_features,
        'nxt_features': batched_nxt_features,
        'cur_labels': batched_cur_labels,
        'nxt_labels': batched_nxt_labels,
        'not_final': not_final
    }
