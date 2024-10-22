export PYTHONPATH=/home/jmabq/projects/VirtualStainingSOTA
nohup python predict.py \
    --dataset_name cuhk \
    --data_root /jhcnas3/VirtualStaining/Patches/HE2PAS_256 \
    --checkpoint_path /jhcnas1/jmabq/virtual_staining_sota/Our/cuhk/model/netG_epoch_4.pth \
    --results_dir /jhcnas1/jmabq/virtual_staining_sota/Our/cuhk/prediction \
    --crop_size 128 \
    --noise 0 \
    --batch_size 32 \
    --gpu_id 7 > cuhk.log 2>&1 &