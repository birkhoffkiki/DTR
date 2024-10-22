export PYTHONPATH=/jhcnas5/jmabq/code/Our/datasets
nohup python train.py \
    --dataset_name cuhk \
    --data_root /jhcnas3/VirtualStaining/Patches/HE2PAS_256 \
    --crop_size 128 \
    --noise 2 \
    --n_epochs 20 \
    --lr_G 5e-5 \
    --GAN_weight 0.1 \
    --save_per_epoch 1 \
    --continue_weight_id 4 \
    --batch_size 32 \
    --gpu_id 4 \
    --checkpoint_dir /jhcnas1/jmabq/virtual_staining_sota/Our/cuhk > train_cuhk.log 2>&1 &
