import os

class CFG:
    # Debugging
    debug = False
    
    # Plot training history
    training_plot = True
    
    # Weights and Biases logging
    wandb = True
    competition   = 'birdclef-2023'
    _wandb_kernel = 'hirokin1999'
    
    # Experiment name and comment
    exp_name = 'baseline'
    comment = 'exp29|add union(20,21,22,xeno)|loss=bce,pos_weight=InverseFrequency|fine-tuning(exp-22)|2022_1st_like|2022_3rd_like_label'
    
    # Notebook link
    notebook_link = 'https://www.kaggle.com/awsaf49/birdclef23-effnet-fsr-cutmixup-train/edit'
    
    # PATH
    BASE_PATH = '/kaggle/input/birdclef-2023'
    BACKNOISE_BASE_PATH = '/kaggle/input/birdclef-2023-dataset/birdclef2021-background-noise_wav'
    
    # 20/21/22/xeno
    BASE_DIR2 = '/kaggle/input/birdclef-2023-dataset'
    BASE_PATH_20 = f'{BASE_DIR2}/birdsong-recognition'
    BASE_PATH_21 = f'{BASE_DIR2}/birdclef-2021'
    BASE_PATH_22 = f'{BASE_DIR2}/birdclef-2022'
    BASE_PATH_xam = f'{BASE_DIR2}/xeno-canto-bird-recordings-extended-a-m'
    BASE_PATH_xnz = f'{BASE_DIR2}/xeno-canto-bird-recordings-extended-n-z'
    
    # Verbosity level
    verbose = 0
    
    # Device and random seed
    device = 'cuda:0'
    seed = 42
    
    # Input image size and batch size
    img_size = [128, 1001]
    batch_size = 64
    upsample_thr = 50 # min sample of each class (upsample)
    cv_filter = True # always keeps low sample data in train
    # Inference batch size, test time augmentation, and drop remainder
    valid_bs = 128
    test_bs = 2
    tta = 1
    drop_remainder = True
    
    # Number of epochs, model name, and number of folds
    epochs = 20
    model_name = 'tf_efficientnet_b1_ns'
    in_chans = 1
    num_fold = 5
    
    # Selected folds for training and evaluation
    selected_folds = [0, 1, 2, 3, 4]

    # Pretraining, neck features, and final activation function
    pretrain = True
    pretrained_model_path = '/kaggle/working/multi_label/exp22_primary_label_2022_1st_like'
    inference_model_path = 'input/fold-0.pth'
    #final_act = 'softmax'
    
    # Learning rate, optimizer, and scheduler
    lr = 1e-4
    lr_min = 1e-5
    warmup_t = 5
    warmup_lr_init = 5e-5
    
    scheduler = 'cos'
    optimizer = 'AdamW' # AdamW, RectifiedAdam, Adam
    
    # Loss function and label smoothing
    loss = 'CCE' # BCE, CCE
    label_smoothing = 0.05 # label smoothing
    
    # Audio duration, sample rate, and length
    duration = 10 # second
    test_duration = 5
    sample_rate = 32000
    audio_len = duration*sample_rate
    
    # STFT parameters
    n_mels = 128
    n_fft = 2028
    hop_length = 320
    fmin = 50
    fmax = 14000
    top_db=None
    normalize = True
    
    
    
    # Data augmentation parameters
    augment=True
    
    # Spec augment
    spec_augment_prob = 0.80
    
    mixup_prob = 0.65
    mixup_alpha = 0.5
    
    cutmix_prob = 0.0
    cutmix_alpha = 0.5
    
    mask_prob = 0.5
    freq_mask = 10
    time_mask = 20
    # random crop aug
    random_crop = False

    # Audio Augmentation Settings
    audio_augment_prob = 0.5
    
    timeshift_prob = 0.0
    
    gn_prob = 0.5

    # Data Preprocessing Settings
    class_names = sorted(os.listdir('/kaggle/input/birdclef-2023/train_audio/'))
    num_classes = len(class_names)
    class_labels = list(range(num_classes))
    label2name = dict(zip(class_labels, class_names))
    name2label = {v:k for k,v in label2name.items()}

    # Class Labels for BirdCLEF 21 & 22
    class_names_pre = sorted(set(os.listdir(f'{BASE_PATH_21}/train_short_audio/')
                       +os.listdir(f'{BASE_PATH_22}/train_audio/')
                       +os.listdir(f'{BASE_PATH_20}/train_audio/')))
    num_classes_pre = len(class_names_pre)
    class_labels_pre = list(range(num_classes_pre))
    label2name_pre = dict(zip(class_labels_pre, class_names_pre))
    name2label_pre = {v:k for k,v in label2name_pre.items()}

    # Training Settings
    target_col = ['target']
    tab_cols = ['filename']
    monitor = 'auc'