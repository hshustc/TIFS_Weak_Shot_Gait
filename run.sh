# Adaptive Knowledge Transfer for Weak-Shot Gait Recognition

# WS-Upper
export MODEL=AMP_DDP_casia_b_noiserate0.00_rt128_train_base_bin16_cbs8x16 && \
    python -m torch.distributed.launch --nproc_per_node=8 --master_port 29502 train.py \
    --dataset CASIA-B --resolution 128 --dataset_path /dev/shm/Dataset/casia_b_nr0.00_seed2022/silhouettes_cut128_pkl_nr0.00_seed2022/ \
    --pid_fname partition/CASIA_B_nr0.00_seed2022.npy --noise_split False \
    --clean_batch_size 8 16 --noise_batch_size 0 16 \
    --milestones 10000 20000 30000 --total_iter 35000 --warmup False \
    --more_channels False --bin_num 16 --hidden_dim 256 \
    --encoder_triplet_weight 1.0 --encoder_triplet_margin 0.2 --encoder_triplet_weakshot False \
    --model_name $MODEL --gpu 0,1,2,3,4,5,6,7 \
    --AMP True --DDP True \
    2>&1 | tee $MODEL.log

# WS-Base
export MODEL=AMP_DDP_casia_b_noiserate0.50_rt128_train_base_bin16_cbs8x16 && \
    python -m torch.distributed.launch --nproc_per_node=8 --master_port 29502 train.py \
    --dataset CASIA-B --resolution 128 --dataset_path /dev/shm/Dataset/casia_b_nr0.50_seed2022/silhouettes_cut128_pkl_nr0.50_seed2022/ \
    --pid_fname partition/CASIA_B_nr0.50_seed2022.npy --noise_split False \
    --clean_batch_size 8 16 --noise_batch_size 0 16 \
    --milestones 10000 20000 30000 --total_iter 35000 --warmup False \
    --more_channels False --bin_num 16 --hidden_dim 256 \
    --encoder_triplet_weight 1.0 --encoder_triplet_margin 0.2 --encoder_triplet_weakshot False \
    --model_name $MODEL --gpu 0,1,2,3,4,5,6,7 \
    --AMP True --DDP True \
    2>&1 | tee $MODEL.log

# AKT-Base
export MODEL=AMP_DDP_casia_b_noiserate0.50_rt128_train_base_bin16_cbs4x16_nbs8x8 && \
    python -m torch.distributed.launch --nproc_per_node=8 --master_port 29502 train.py \
    --dataset CASIA-B --resolution 128 --dataset_path /dev/shm/Dataset/casia_b_nr0.50_seed2022/silhouettes_cut128_pkl_nr0.50_seed2022/ \
    --pid_fname partition/CASIA_B_nr0.50_seed2022.npy --noise_split True \
    --clean_batch_size 4 16 --noise_batch_size 8 8 \
    --milestones 10000 20000 30000 --total_iter 35000 --warmup False \
    --more_channels False --bin_num 16 --hidden_dim 256 \
    --encoder_triplet_weight 1.0 --encoder_triplet_margin 0.2 --encoder_triplet_weakshot True \
    --model_name $MODEL --gpu 0,1,2,3,4,5,6,7 \
    --AMP True --DDP True \
    2>&1 | tee $MODEL.log

# AKT-KD
export MODEL=AMP_DDP_casia_b_noiserate0.50_rt128_train_base_bin16_cbs4x16_nbs8x8_temp16_v4 && \
    python -m torch.distributed.launch --nproc_per_node=8 --master_port 29502 train.py \
    --dataset CASIA-B --resolution 128 --dataset_path /dev/shm/Dataset/casia_b_nr0.50_seed2022/silhouettes_cut128_pkl_nr0.50_seed2022/ \
    --pid_fname partition/CASIA_B_nr0.50_seed2022.npy --noise_split True \
    --clean_batch_size 4 16 --noise_batch_size 8 8 \
    --milestones 10000 20000 30000 --total_iter 35000 --warmup False \
    --more_channels False --bin_num 16 --hidden_dim 256 \
    --encoder_triplet_weight 1.0 --encoder_triplet_margin 0.2 --encoder_triplet_weakshot True \
    --encoder_infonce_weight 0.1 --encoder_infonce_temperature 16.0 --encoder_infonce_weakshot True \
    --encoder_infonce_poshard True --encoder_infonce_neghard True \
    --model_name $MODEL --gpu 0,1,2,3,4,5,6,7 \
    --AMP True --DDP True \
    2>&1 | tee $MODEL.log

# AKT
export MODEL=AMP_DDP_casia_b_noiserate0.50_rt128_train_base_bin16_cbs4x16_nbs8x8_temp16_v4_adacont_lrv1_nhardtempv1 && \
    python -m torch.distributed.launch --nproc_per_node=8 --master_port 29502 train.py \
    --dataset CASIA-B --resolution 128 --dataset_path /dev/shm/Dataset/casia_b_nr0.50_seed2022/silhouettes_cut128_pkl_nr0.50_seed2022/ \
    --pid_fname partition/CASIA_B_nr0.50_seed2022.npy --noise_split True \
    --clean_batch_size 4 16 --noise_batch_size 8 8 \
    --noise_hard True --noise_momentum 0.9 --noise_temperature 1.0 \
    --milestones 10000 20000 30000 --total_iter 35000 --warmup False \
    --more_channels False --bin_num 16 --hidden_dim 256 \
    --encoder_triplet_weight 1.0 --encoder_triplet_margin 0.2 --encoder_triplet_weakshot True \
    --encoder_infonce_weight 0.1 --encoder_infonce_temperature 16.0 --encoder_infonce_weakshot True \
    --encoder_infonce_poshard True --encoder_infonce_neghard True \
    --encoder_adacont_weight 0.5 --encoder_adacont_posthres 0.5 --encoder_adacont_negthres 0.1 \
    --init_model AMP_DDP_casia_b_noiserate0.50_rt128_train_base_bin16_cbs4x16_nbs8x8_temp16_v4_CASIA-B_73_False-35000-encoder.ptm \
    --lr 0.001 --milestones 10000 --total_iter 15000 \
    --model_name $MODEL --gpu 0,1,2,3,4,5,6,7 \
    --AMP True --DDP True \
    2>&1 | tee $MODEL.log