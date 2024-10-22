export PYTHONPATH=/jhcnas5/jmabq/code/Our/datasets
nohup python train.py \
    --dataset_name aperio \
    --data_root /home/jmabq/data/aperio_hamamatsu \
    --crop_size 128 \
    --noise 2 \
    --n_epochs 200 \
    --lr_G 5e-5 \
    --GAN_weight 0.1 \
    --save_per_epoch 1 \
    --batch_size 32 \
    --gpu_id 7 \
    --checkpoint_dir /jhcnas1/jmabq/virtual_staining_sota/Our/aperio > train_aperio.log 2>&1 &

