export PYTHONPATH=/home/jmabq/projects/VirtualStainingSOTA
nohup python predict.py \
    --dataset_name hemit \
    --data_root /home/jmabq/data/HEMIT \
    --checkpoint_path /jhcnas1/jmabq/virtual_staining_sota/Our/hemit/model/netG_epoch_72.pth \
    --results_dir /jhcnas1/jmabq/virtual_staining_sota/Our/hemit/model/prediction \
    --crop_size 512 \
    --noise 0 \
    --batch_size 4 \
    --gpu_id 7 > hemit.log 2>&1 &