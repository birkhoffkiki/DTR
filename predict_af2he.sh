export PYTHONPATH=/home/jmabq/projects/VirtualStainingSOTA
nohup python predict.py \
    --dataset_name af2he \
    --data_root /home/jmabq/data/af2he_uniform \
    --checkpoint_path /jhcnas1/jmabq/virtual_staining_sota/Our/af2he/model/netG_epoch_100.pth \
    --results_dir /jhcnas1/jmabq/virtual_staining_sota/Our/af2he/prediction \
    --crop_size 128 \
    --noise 0 \
    --batch_size 16 \
    --gpu_id 6 > af2he.log 2>&1 &