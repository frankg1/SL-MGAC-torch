#!/bin/bash

# Set common parameters
TXT_FILE_PATH="rl_model_train_data_for_torch_1011_cleaned_train.txt"
BATCH_SIZE=2048
EPOCH_NUM=500
MODEL_SAVE_DIR="save_weights/"
ENABLE_SAVE_MODEL=true
ENABLE_TB=true

# Run each training job with different model names in the background
nohup python3 train.py --txt_file_path $TXT_FILE_PATH --batch_size $BATCH_SIZE --epoch_num $EPOCH_NUM --model_save_dir $MODEL_SAVE_DIR --enable_save_model $ENABLE_SAVE_MODEL --model_name Model --enable_tb $ENABLE_TB --device_name 1 >> nohup.out 2>&1 &
nohup python3 train.py --txt_file_path $TXT_FILE_PATH --batch_size $BATCH_SIZE --epoch_num $EPOCH_NUM --model_save_dir $MODEL_SAVE_DIR --enable_save_model $ENABLE_SAVE_MODEL --model_name ModelK1 --enable_tb $ENABLE_TB --device_name 1 >> nohup.out 2>&1 &
nohup python3 train.py --txt_file_path $TXT_FILE_PATH --batch_size $BATCH_SIZE --epoch_num $EPOCH_NUM --model_save_dir $MODEL_SAVE_DIR --enable_save_model $ENABLE_SAVE_MODEL --model_name ModelK2 --enable_tb $ENABLE_TB --device_name 1 >> nohup.out 2>&1 &
nohup python3 train.py --txt_file_path $TXT_FILE_PATH --batch_size $BATCH_SIZE --epoch_num $EPOCH_NUM --model_save_dir $MODEL_SAVE_DIR --enable_save_model $ENABLE_SAVE_MODEL --model_name ModelK3 --enable_tb $ENABLE_TB --device_name 1 >> nohup.out 2>&1 &
nohup python3 train.py --txt_file_path $TXT_FILE_PATH --batch_size $BATCH_SIZE --epoch_num $EPOCH_NUM --model_save_dir $MODEL_SAVE_DIR --enable_save_model $ENABLE_SAVE_MODEL --model_name ModelK4 --enable_tb $ENABLE_TB --device_name 1 >> nohup.out 2>&1 &
nohup python3 train.py --txt_file_path $TXT_FILE_PATH --batch_size $BATCH_SIZE --epoch_num $EPOCH_NUM --model_save_dir $MODEL_SAVE_DIR --enable_save_model $ENABLE_SAVE_MODEL --model_name ModelK5 --enable_tb $ENABLE_TB --device_name 1 >> nohup.out 2>&1 &
nohup python3 train.py --txt_file_path $TXT_FILE_PATH --batch_size $BATCH_SIZE --epoch_num $EPOCH_NUM --model_save_dir $MODEL_SAVE_DIR --enable_save_model $ENABLE_SAVE_MODEL --model_name Model_Gr --enable_tb $ENABLE_TB --device_name 1 >> nohup.out 2>&1 &
nohup python3 train.py --txt_file_path $TXT_FILE_PATH --batch_size $BATCH_SIZE --epoch_num $EPOCH_NUM --model_save_dir $MODEL_SAVE_DIR --enable_save_model $ENABLE_SAVE_MODEL --model_name Model_GrBu --enable_tb $ENABLE_TB  --device_name 1 >> nohup.out 2>&1 &
nohup python3 train.py --txt_file_path $TXT_FILE_PATH --batch_size $BATCH_SIZE --epoch_num $EPOCH_NUM --model_save_dir $MODEL_SAVE_DIR --enable_save_model $ENABLE_SAVE_MODEL --model_name Model_GrBuSl --enable_tb $ENABLE_TB  --device_name 1 >> nohup.out 2>&1 &
nohup python3 train.py --txt_file_path $TXT_FILE_PATH --batch_size $BATCH_SIZE --epoch_num $EPOCH_NUM --model_save_dir $MODEL_SAVE_DIR --enable_save_model $ENABLE_SAVE_MODEL --model_name Model_LN --enable_tb $ENABLE_TB  --device_name 1 >> nohup.out 2>&1 &
nohup python3 train.py --txt_file_path $TXT_FILE_PATH --batch_size $BATCH_SIZE --epoch_num $EPOCH_NUM --model_save_dir $MODEL_SAVE_DIR --enable_save_model $ENABLE_SAVE_MODEL --model_name Model_SG --enable_tb $ENABLE_TB  --device_name 1 >> nohup.out 2>&1 &
nohup python3 train.py --txt_file_path $TXT_FILE_PATH --batch_size $BATCH_SIZE --epoch_num $EPOCH_NUM --model_save_dir $MODEL_SAVE_DIR --enable_save_model $ENABLE_SAVE_MODEL --model_name Model_SQ --enable_tb $ENABLE_TB  --device_name 1 >> nohup.out 2>&1 &
nohup python3 train.py --txt_file_path $TXT_FILE_PATH --batch_size $BATCH_SIZE --epoch_num $EPOCH_NUM --model_save_dir $MODEL_SAVE_DIR --enable_save_model $ENABLE_SAVE_MODEL --model_name Model_SGQ --enable_tb $ENABLE_TB --device_name 1 >> nohup.out 2>&1 &
nohup python3 train.py --txt_file_path $TXT_FILE_PATH --batch_size $BATCH_SIZE --epoch_num $EPOCH_NUM --model_save_dir $MODEL_SAVE_DIR --enable_save_model $ENABLE_SAVE_MODEL --model_name Model_NotQ --enable_tb $ENABLE_TB --device_name 1 >> nohup.out 2>&1 &
echo "All training processes are running in the background. Check nohup.out for output."
