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
    comment = 'EfficientNetB1|128x1001|t=8s|lr=1e-3|act=sigmoid|fine-tuning(exp-10)|2022_1st_like'
    
    # Notebook link
    notebook_link = 'https://www.kaggle.com/awsaf49/birdclef23-effnet-fsr-cutmixup-train/edit'
    
    # PATH
    BASE_PATH = '/kaggle/input/birdclef-2023'
    BACKNOISE_BASE_PATH = '/kaggle/input/birdclef-2023-dataset/birdclef2021-background-noise_wav'
    
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
    epochs = 50
    model_name = 'tf_efficientnet_b1_ns'
    in_chans = 1
    num_fold = 5
    
    # Selected folds for training and evaluation
    selected_folds = [0, 1, 2, 3, 4]

    # Pretraining, neck features, and final activation function
    pretrain = True
    pretrained_model_path = '/kaggle/working/WSL/exp10_softmax_model'
    inference_model_path = 'input/fold-0.pth'
    #final_act = 'softmax'
    
    # Learning rate, optimizer, and scheduler
    lr = 1e-3
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

    # Training Settings
    target_col = ['target']
    tab_cols = ['filename']
    monitor = 'auc'