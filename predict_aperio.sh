export PYTHONPATH=/home/jmabq/projects/VirtualStainingSOTA
nohup python predict.py \
    --dataset_name aperio \
    --data_root /home/jmabq/data/aperio_hamamatsu \
    --checkpoint_path /jhcnas1/jmabq/virtual_staining_sota/Our/aperio/model/netG_epoch_97.pth \
    --results_dir /jhcnas1/jmabq/virtual_staining_sota/Our/aperio/model/prediction \
    --crop_size 128 \
    --noise 0 \
    --batch_size 16 \
    --gpu_id 2 > aperio.log 2>&1 &