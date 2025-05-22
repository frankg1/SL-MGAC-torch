#!/bin/bash

# Set common parameters
TXT_FILE_PATH="rl_model_train_data_for_torch_1011_cleaned_train.txt"
BATCH_SIZE=2048
EPOCH_NUM=500
MODEL_SAVE_DIR="save_weights/"
ENABLE_SAVE_MODEL=true
ENABLE_TB=true
DEVICE_NAME=0
# Run each training job with different model names in the background
nohup python3 train_lys.py --txt_file_path $TXT_FILE_PATH --batch_size $BATCH_SIZE --epoch_num $EPOCH_NUM --model_save_dir $MODEL_SAVE_DIR --enable_save_model $ENABLE_SAVE_MODEL --model_name BCQ --enable_tb $ENABLE_TB --device_name $DEVICE_NAME >> nohup.out 2>&1 &
nohup python3 train_lys.py --txt_file_path $TXT_FILE_PATH --batch_size $BATCH_SIZE --epoch_num $EPOCH_NUM --model_save_dir $MODEL_SAVE_DIR --enable_save_model $ENABLE_SAVE_MODEL --model_name DQN --enable_tb $ENABLE_TB --device_name $DEVICE_NAME >> nohup.out 2>&1 &
nohup python3 train_lys.py --txt_file_path $TXT_FILE_PATH --batch_size $BATCH_SIZE --epoch_num $EPOCH_NUM --model_save_dir $MODEL_SAVE_DIR --enable_save_model $ENABLE_SAVE_MODEL --model_name SAC --enable_tb $ENABLE_TB --device_name $DEVICE_NAME >> nohup.out 2>&1 &
nohup python3 train_lys.py --txt_file_path $TXT_FILE_PATH --batch_size $BATCH_SIZE --epoch_num $EPOCH_NUM --model_save_dir $MODEL_SAVE_DIR --enable_save_model $ENABLE_SAVE_MODEL --model_name TD3 --enable_tb $ENABLE_TB --device_name $DEVICE_NAME >> nohup.out 2>&1 &
nohup python3 train_lys.py --txt_file_path $TXT_FILE_PATH --batch_size $BATCH_SIZE --epoch_num $EPOCH_NUM --model_save_dir $MODEL_SAVE_DIR --enable_save_model $ENABLE_SAVE_MODEL --model_name TD3_BC --enable_tb $ENABLE_TB --device_name $DEVICE_NAME >> nohup.out 2>&1 &
nohup python3 train_lys.py --txt_file_path $TXT_FILE_PATH --batch_size $BATCH_SIZE --epoch_num $EPOCH_NUM --model_save_dir $MODEL_SAVE_DIR --enable_save_model $ENABLE_SAVE_MODEL --model_name IQL --enable_tb $ENABLE_TB --device_name $DEVICE_NAME >> nohup.out 2>&1 &
nohup python3 train_lys.py --txt_file_path $TXT_FILE_PATH --batch_size $BATCH_SIZE --epoch_num $EPOCH_NUM --model_save_dir $MODEL_SAVE_DIR --enable_save_model $ENABLE_SAVE_MODEL --model_name L2R --enable_tb $ENABLE_TB --device_name $DEVICE_NAME >> nohup.out 2>&1 &
echo "All training processes are running in the background. Check nohup.out for output."

#python3 train_lys.py --txt_file_path rl_model_train_data_for_torch_1011_cleaned_mini.txt --batch_size 4 --epoch_num 2 --model_save_dir save_weights/ --enable_save_model false --model_name BCQ 
