export PYTHONPATH=/jhcnas5/jmabq/code/Our/datasets
nohup python train.py \
    --dataset_name hemit \
    --data_root /home/jmabq/data/HEMIT \
    --crop_size 512 \
    --noise 2 \
    --n_epochs 100 \
    --save_per_epoch 5 \
    --lr_G 1e-5 \
    --batch_size 2 \
    --gpu_id 3 \
    --checkpoint_dir /jhcnas1/jmabq/virtual_staining_sota/Our/hemit > train_hemit.log 2>&1 &
