export PYTHONPATH=/jhcnas5/jmabq/code/Our/datasets
nohup python train.py \
    --dataset_name af2he \
    --data_root /home/jmabq/data/af2he_uniform \
    --crop_size 128 \
    --noise 2 \
    --n_epochs 200 \
    --lr_G 5e-5 \
    --GAN_weight 0.1 \
    --save_per_epoch 1 \
    --batch_size 32 \
    --gpu_id 5 \
    --checkpoint_dir /jhcnas1/jmabq/virtual_staining_sota/Our/af2he > train_af2he.log 2>&1 &