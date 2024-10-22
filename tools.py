class ConfigObj:
    # reg config
    init_to_identity = True
    # network and training
    batch_size = 8
    image_size = 128
    n_channels = 64
    n_blocks = 5
    input_ch_num = 4
    output_ch_num = 3

    checkpoint = 3
    lamda = 10  # L1 weight
    nu = 0.2  # ssim weight
    alpha = 0.5  # dis weight
    lr_G = 1e-4
    lr_D = 1e-5
    lambda_l1 = 10
    weight_smooth = 20
    weight_same_mesh = 5

    # data loading
    n_threads = 4
    crop_size = 128
    max_epochs = 100

    epoch_len = 500
    data_queue_len = 10000
    patch_per_tile = 10
    color_space = "RGB"

    train_images_dir = r"train_data/train/target"
    valid_images_dir = r"train_data/valid/target"
