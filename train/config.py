config = {

    #
    # This is for experiment.
    # It often results in NAN value during training.
    #
    'mixed_precision': False,
    
    #
    # This is for experiment and it is used for fine-tuning.
    #
    'enable_prune': False,
    'prune_params':{
        'initial_sparsity': 0.3,
        'final_sparsity': 0.5,
        'begin_step': 0,
        'end_step': 30000
    },

    #
    # This is for experiment and it is used for fine-tuning.
    #
    'enable_quant_aware' : False,

    #
    # Save trained model named with 'model_name'.h5.
    # The best model at each epoch is saved to the folder ./checkpoint/'model_name'.
    #
    'model_name': 'ResNet50_centerloss_yourdataset_4000',

    #
    # Restore weights of backbone network.
    # It restores only weights of backbone network.
    #
    # 'saved_backbone': '',
    'saved_backbone': './checkpoints/ResNet50_arcface_251001_2/best_full_backbone.keras',

    #
    # The checkpoint option is different from saved_backbone option.
    # It restores the entire weights of a custom model.
    # So it overrides the weights of saved_backbone with the weights in checkpoint if you feed both options.
    # The path should indicate a directory containing checkpoint files.
    #
    # 'checkpoint': './checkpoints/ResNet50_adaface_251001_2/ckpt/',
    'checkpoint': '',

    'batch_size' : 256,

    #
    # It is for training with large batch size on a limited GPU memory.
    # It accumulates gradients for 'num_grad_accum' times, then applies accumulated gradients.
    # The total batch size is 'batch_size' * 'num_grad_accum'.
    # ex) 'num_grad_accum' = 4 and 'batch_size' = 256, then total batch size is 1024.
    #
    'num_grad_accum': 4,
    'shape' : [112, 112, 3],

    #
    # Choose one of below: 
    # 1. MobileNetV2
    # 2. MobileNetV3
    # 3. EfficientNetB3
    # 4. ResNet50
    #
    'model' : 'ResNet50',
    'embedding_dim': 512,

    #
    # 1. SoftmaxCenter (known as Center Loss)
    # 2. AngularMargin (known as ArcFace)
    # 3. GroupAware (known as GroupFace)
    # 4. CosFace (known as AM-LFC)
    # 5. AdaFace (known as AM-LFS)
    #
    'loss': 'AngularMargin',
    'loss_param':{
        'SoftmaxCenter':{
            'scale': 30,
            'center_loss_weight': 1e-3,
        },
        'AngularMargin':{
            'scale': 60,
            'margin': 0.5
        },
        'GroupAware':{
            'scale': 60,
            'margin': 0.5,
            'num_groups': 4,
            'intermidiate_dim': 256,
            'group_loss_weight': 0.1
        },
        'CosFace':{
            'scale': 60,
            'margin': 0.35
        },
        'AdaFace':{
            'scale': 75,
            'margin': 0.5,
            'h': 0.4
        }
    },

    'eval':{
        'metric': 'cos',
        'recall': [1]
    },

    #
    # There are two options.
    #  1. Adam
    #  2. AdamW
    #  3. SGD with momentum=0.9 and nesterov=True
    #
    'optimizer' : 'AdamW',
    'epoch' : 40,

    #
    # initial learning rate.
    #
    'lr' : 1e-4,

    #
    # SGD optimizer settings (optimized for large dataset)
    #
    'optimizer_params': {
        'momentum': 0.9,
        'nesterov': True,
        'weight_decay': 5e-4
    },

    #
    # learning rate scheduling (Cosine Annealing + Warmup)
    #
    'lr_scheduler': {
        'type': 'cosine',  # cosine annealing with warmup
        'warmup_epochs': 10,  # warmup 10 epochs
        'min_lr': 1e-6,  # minimum learning rate
        'T_max': None  # total epochs (automatically calculated)
    },

    #
    # training dataset generated from generate_tfrecord/main.py
    # See README.md
    #
    'train_file': 'your_train.tfrecord',
    'test_files': ['your_test.tfrecord'],

    #
    # Set maximum face ID in 'tfrecord_file'.
    #
    'num_identity': 111182,

    # global shuffle option - completely random shuffle
    'global_shuffle': {
        'enabled': False,
        'seed': 42
    },

    # shuffle buffer size setting
    'shuffle_buffer_size': 10000,  # shuffle buffer size
    # options:
    # - None: automatically set to batch_size * 1000 (256 * 1000 = 256,000)
    # - integer: directly specify (e.g. 10000, 100000, 500000)
    # - 'auto': 1% of dataset size (maximum 500,000)
    #
    # usage examples:
    # 'shuffle_buffer_size': None,        # automatically set (recommended)
    # 'shuffle_buffer_size': 100000,      # medium size
    # 'shuffle_buffer_size': 500000,      # large dataset
    # 'shuffle_buffer_size': 'auto',      # 1% of dataset size
    #
    # memory usage reference:
    # - 100,000: approximately 1.5GB GPU memory usage
    # - 500,000: approximately 7.5GB GPU memory usage
    # - 1,000,000: approximately 15GB GPU memory usage

    # throughput monitoring: per-epoch sample count (if <=0, automatically count from actual dataset size)
    'total_samples': 6122484,
}
