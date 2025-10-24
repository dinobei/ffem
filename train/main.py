import argparse
import importlib
import copy
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import train.input_pipeline as input_pipeline
import train.config
from train.callbacks import LogCallback, CSVLoggerCallback, BackboneCheckpoint
from train.callbacks import WeightsCheckpoint
from train.callbacks import RecallCallback
from train.utils import apply_pruning
from train.utils import apply_quantization_aware
import net_arch.models
import train.blocks
from train.custom_models.softmax_center_model import SoftmaxCenterModel
from train.custom_models.angular_margin_model import AngularMarginModel
from train.custom_models.group_aware_model import GroupAwareModel
from train.custom_models.cosface_model import CosFaceModel
from train.custom_models.adaface_model import AdaFaceModel

import tensorflow as tf
import numpy as np

def count_dataset_samples(dataset_path):
    """calculate actual samples in dataset (support list of paths)"""
    try:
        import tensorflow as tf
        import os
        from tqdm import tqdm

        def _count_one(path):
            try:
                print(f"🔍 Counting samples in: {path}")
                try:
                    file_size = os.path.getsize(path)
                    print(f"📁 File size: {file_size / (1024*1024):.1f} MB")
                except Exception:
                    pass
                cnt = 0
                dataset = tf.data.TFRecordDataset(path)
                with tqdm(desc=f"Counting {os.path.basename(path)}", unit=" samples") as pbar:
                    for _ in dataset:
                        cnt += 1
                        pbar.update(1)
                print(f"📊 Actual dataset size: {cnt:,} samples")
                return cnt
            except Exception as e:
                print(f"⚠️  Could not count dataset samples in {path}: {e}")
                return 0

        if isinstance(dataset_path, (list, tuple)):
            total = 0
            for p in dataset_path:
                total += _count_one(p)
            print(f"📊 Combined dataset size: {total:,} samples")
            return total
        else:
            return _count_one(dataset_path)
    except Exception as e:
        print(f"⚠️  Could not count dataset samples: {e}")
        return None

def get_actual_dataset_size(config):
    """get actual dataset size from config['total_samples'] or count train_files"""
    # 1) honor manual per-epoch total if provided (>0)
    manual = config.get('total_samples', None)
    if manual is not None and manual > 0:
        return int(manual)

    # 2) otherwise, count from training files
    train_files = config.get('train_files')
    if isinstance(train_files, (list, tuple)) and len(train_files) > 0:
        actual_samples = count_dataset_samples(train_files)
        if actual_samples and actual_samples > 0:
            return actual_samples
    elif isinstance(train_files, str) and train_files:
        actual_samples = count_dataset_samples(train_files)
        if actual_samples and actual_samples > 0:
            return actual_samples
    # default value (for small test dataset)
    default_samples = 10000
    print(f"⚠️  Using default dataset size: {default_samples:,} samples per epoch")
    return default_samples


def build_dataset(config):
    train_ds, test_ds_dict = input_pipeline.make_tfdataset(
        config['train_files'],
        config['test_files'],
        config['batch_size'],
        config['shape'][:2],
        config.get('shuffle_buffer_size', None),
        config.get('global_shuffle', None),
    )
    return train_ds, test_ds_dict


def build_backbone_model(config):
    is_pretrained = False
    # if ckpt exists, skip backbone file (ckpt contains all backbone + head)
    checkpoint_dir = config.get('checkpoint', None)
    
    # if ckpt exists, skip backbone file
    if checkpoint_dir and tf.train.latest_checkpoint(checkpoint_dir) is not None:
        print('\n---------------- Skip Backbone Load (ckpt will restore all) ----------------\n')
        print('checkpoint directory has ckpt files, skipping saved_backbone load')
        print('\n----------------------------------------------------------------------------\n')
        net = net_arch.models.get_model(config['model'], config['shape'])
    elif os.path.exists(config['saved_backbone']):
        net = tf.keras.models.load_model(config['saved_backbone'])
        print('\n---------------- Restore Backbone Network ----------------\n')
        print(config['saved_backbone'])
        print('\n----------------------------------------------------------\n')
        is_pretrained = True
    else:
        net = net_arch.models.get_model(config['model'], config['shape'])

    return net, is_pretrained


def build_model(config):
    net, is_pretrained = build_backbone_model(config)
    if config['enable_prune']:
        net = apply_pruning(
            net, config['prune_params'], 1 if is_pretrained else None)

    param = copy.deepcopy(config['loss_param'][config['loss']])
    dummy_x = np.zeros([config['batch_size']] + config['shape'])
    dummy_y = np.zeros([config['batch_size']]+[config['num_identity']])
    model = None
    if config['loss'] == 'SoftmaxCenter':
        param['n_classes'] = config['num_identity']
        param['embedding_dim'] = config['embedding_dim']
        model = SoftmaxCenterModel(net, **param, name=config['model_name'])
    elif config['loss'] == 'AngularMargin':
        param['n_classes'] = config['num_identity']
        param['embedding_dim'] = config['embedding_dim']
        model = AngularMarginModel(net, **param, name=config['model_name'])
    elif config['loss'] == 'GroupAware':
        param['n_classes'] = config['num_identity']
        param['instance_dim'] = config['embedding_dim']
        model = GroupAwareModel(net, **param, name=config['model_name'])
    elif config['loss'] == 'CosFace':
        param['n_classes'] = config['num_identity']
        param['embedding_dim'] = config['embedding_dim']
        model = CosFaceModel(net, **param, name=config['model_name'])
    elif config['loss'] == 'AdaFace':
        param['n_classes'] = config['num_identity']
        param['embedding_dim'] = config['embedding_dim']
        model = AdaFaceModel(net, **param, name=config['model_name'])
    else:
        raise Exception('The loss ({}) is not supported.'.format(config['loss']))

    # Do pre-compile
    # This is a tensorflow bug.
    # After restoring a checkpoint without pre-compile, 
    #  learning rate is overridden with a checkpoint, when it compiles with a new optimizer.
    model.compile()
    
    # restore weights: if ckpt exists, restore full model, otherwise restore backbone only
    restore_latest_checkpoint(model, config['checkpoint'])
    if config['enable_quant_aware']:
        model.backbone = apply_quantization_aware(model.backbone, None)
    model([dummy_x, dummy_y], training=True) # dummy call for building model
    return model


def build_callbacks(config, test_ds_dict):
    """create callbacks - include advanced metrics"""
    log_dir = os.path.join('logs', config['model_name'])
    callback_list = []
    metric = config['eval']['metric']
    recall_topk = config['eval']['recall']
    # set advanced metrics calculation interval (1 epoch)
    advanced_metrics_interval = 1
    recall_eval = RecallCallback(test_ds_dict, recall_topk, metric, log_dir, advanced_metrics_interval, config['model_name'])
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='recall@1', factor=0.5, mode='max',
        patience=2, min_lr=1e-4, verbose=1)
    
    # checkpoint for inference model (for TFLite conversion)
    from train.callbacks import InferenceModelCheckpoint
    inference_checkpoint = InferenceModelCheckpoint(
        filepath='./checkpoints/{}/best_inference.keras'.format(config['model_name']),
        monitor='recall@1',
        mode='max',
        save_best_only=True,
        verbose=0)  # RecallCallback will print combined log
    
    # checkpoint for full model (for resume training) - also save backbone
    full_checkpoint = BackboneCheckpoint(
        filepath='./checkpoints/{}/best_full.keras'.format(config['model_name']),
        monitor='recall@1',
        mode='max',
        save_best_only=True,
        verbose=0)  # RecallCallback will print combined log

    # checkpoint for weights (for resume training/external evaluation)
    weights_ckpt = WeightsCheckpoint(
        checkpoint_dir='./checkpoints/{}/ckpt'.format(config['model_name']),
        monitor='recall@1',
        mode='max',
        save_best_only=True,
        verbose=0,  # RecallCallback will print combined log
        max_to_keep=5
    )
    
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='recall@1',
        mode='max', patience=100,
        restore_best_weights=True)
    tensorboard_log = LogCallback(log_dir)
    
    # add throughput monitoring callback
    from train.callbacks import ThroughputCallback, CustomProgressBar, NaNMonitorCallback
    
    # get actual dataset size
    samples_per_epoch = get_actual_dataset_size(config)
    total_samples = samples_per_epoch * config['epoch']
    
    # get batch size from config
    batch_size = config['batch_size']
    throughput_monitor = ThroughputCallback(total_samples, log_dir, config)
    custom_progress = CustomProgressBar(samples_per_epoch, batch_size, config['epoch'])
    
    # Progress bar information is printed by CustomProgressBar, so remove it here
    
    # add NaN monitoring callback
    nan_monitor = NaNMonitorCallback(patience=10)
    
    # add CSV logging callback
    csv_logger = CSVLoggerCallback(
        log_dir=os.path.join('checkpoints', config['model_name']),
        config=config,
        test_ds_dict=test_ds_dict,
        top_k=recall_topk,
        metric=metric,
        include_precision=False,  # disable Precision@K calculation (currently inaccurate)
        save_config=True
    )

    callback_list.append(recall_eval)
    callback_list.append(inference_checkpoint)  # save inference model
    callback_list.append(full_checkpoint)       # save full model
    callback_list.append(weights_ckpt)          # save ckpt
    callback_list.append(early_stop)
    callback_list.append(tensorboard_log)
    callback_list.append(throughput_monitor)  # add throughput monitoring
    callback_list.append(custom_progress)  # custom progress bar
    callback_list.append(nan_monitor)  # add NaN monitoring
    callback_list.append(csv_logger)  # add CSV logging
    return callback_list, early_stop


def build_optimizer(config):
    # 학습률 스케줄링 설정
    if config.get('lr_scheduler', {}).get('type') == 'cosine':
        # Cosine Annealing with Warmup
        samples_per_epoch = get_actual_dataset_size(config)
        batch_size = config['batch_size']
        steps_per_epoch = samples_per_epoch // batch_size
        total_steps = config['epoch'] * steps_per_epoch

        warmup_epochs = config['lr_scheduler'].get('warmup_epochs', 10)
        warmup_steps = warmup_epochs * steps_per_epoch
        min_lr = config['lr_scheduler'].get('min_lr', 1e-6)

        print(f"  Cosine Annealing: warmup {warmup_epochs} epochs, min_lr {min_lr}")
        print(f"  Total steps: {total_steps}, Warmup steps: {warmup_steps}")

        # Debug version - let's print the values to understand what's happening
        class CosineAnnealingSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
            def __init__(self, initial_lr, warmup_steps, total_steps, min_lr):
                super().__init__()
                self.initial_lr = initial_lr
                self.warmup_steps = warmup_steps
                self.total_steps = total_steps
                self.min_lr = min_lr

            def __call__(self, step):
                step = tf.cast(step, tf.float32)

                # WARMUP PHASE
                warmup_lr = self.initial_lr * tf.minimum(step / tf.cast(self.warmup_steps, tf.float32), 1.0)

                # COSINE DECAY PHASE
                decay_progress = tf.maximum(step - tf.cast(self.warmup_steps, tf.float32), 0.0)
                decay_steps = tf.cast(self.total_steps - self.warmup_steps, tf.float32)
                progress_ratio = decay_progress / decay_steps
                progress_ratio = tf.clip_by_value(progress_ratio, 0.0, 1.0)

                cosine_decay = 0.5 * (1.0 + tf.cos(tf.constant(3.14159, dtype=tf.float32) * progress_ratio))
                cosine_lr = self.min_lr + (self.initial_lr - self.min_lr) * cosine_decay

                # Use tf.cond for cleaner conditional logic
                return tf.cond(
                    step < tf.cast(self.warmup_steps, tf.float32),
                    lambda: warmup_lr,
                    lambda: cosine_lr
                )

            def get_config(self):
                return {
                    'initial_lr': self.initial_lr,
                    'warmup_steps': self.warmup_steps,
                    'total_steps': self.total_steps,
                    'min_lr': self.min_lr
                }

            @classmethod
            def from_config(cls, config):
                return cls(**config)

        lr = CosineAnnealingSchedule(config['lr'], warmup_steps, total_steps, min_lr)
    else:
        lr = config['lr']

    # optimizer defaults
    optimizer_defaults = {
        'Adam': {'lr': lr},
        'SGD': {
            'learning_rate': lr,
            'momentum': 0.9,
            'nesterov': True
        },
        'AdamW': {
            'learning_rate': lr,
            'weight_decay': 1e-4
        },
    }

    # apply user-defined parameters from config
    optimizer_name = config['optimizer']
    optimizer_params = optimizer_defaults.get(optimizer_name, {}).copy()

    if 'optimizer_params' in config:
        user_params = config['optimizer_params']
        optimizer_params.update(user_params)

    # create optimizer
    opt_list = {
        'Adam': tf.keras.optimizers.Adam,
        'SGD': tf.keras.optimizers.SGD,
        'AdamW': tf.keras.optimizers.AdamW,
    }

    if optimizer_name not in opt_list:
        print(f"{optimizer_name} is not supported.")
        print('Please select one of:', list(opt_list.keys()))
        exit(1)

    optimizer_class = opt_list[optimizer_name]
    optimizer = optimizer_class(**optimizer_params)

    print(f"  Optimizer: {optimizer_name} with params: {optimizer_params}")
    return optimizer


def restore_latest_checkpoint(net, checkpoint_path):
    checkpoint = tf.train.Checkpoint(net)
    latest_path = tf.train.latest_checkpoint(checkpoint_path)
    print('\n---------------- Restore Checkpoint ----------------\n')
    if latest_path is not None:
        print('restore_latest_checkpoint:', latest_path)
        checkpoint.restore(latest_path).expect_partial()
        print('✅ Full model (backbone + head) restored from latest checkpoint')
    else:
        print('Can not find latest checkpoint file:', checkpoint_path)
        print('ℹ️  Using backbone only (head will be randomly initialized)')
    print('\n----------------------------------------------------\n')

def start_training(config):
    # Remove GPU timer warning
    os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
    os.environ['TF_GPU_THREAD_COUNT'] = '1'
    

    if config['mixed_precision']:
        print('---------------- Enabled Mixed Precision ----------------')
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
    tf.get_logger().setLevel('ERROR')
    
    # Build
    train_ds, test_ds_dict = build_dataset(config)
    train_net = build_model(config)
    opt = build_optimizer(config)
    callbacks, early_stop = build_callbacks(config, test_ds_dict)
    train_net.compile(optimizer=opt)
    
    train_net.summary()
    
    # calculate steps_per_epoch (round up to handle last batch)
    samples_per_epoch = get_actual_dataset_size(config)
    steps_per_epoch = (samples_per_epoch + config['batch_size'] - 1) // config['batch_size']
    
    # summarize training configuration (remove duplicates)
    print(f"\n📊 Training Configuration:")
    print(f"  Dataset: {samples_per_epoch:,} samples per epoch")
    print(f"  Batch size: {config['batch_size']} (effective: {config['batch_size'] * config.get('num_grad_accum', 1)})")
    print(f"  Steps per epoch: {steps_per_epoch}")
    print(f"  Total epochs: {config['epoch']}")
    print(f"  Learning rate: {config['lr']}")
    print(f"  Optimizer: {config['optimizer']}")
    print(f"  Loss: {config['loss']}")
    print(f"  Mixed precision: {'Enabled' if config['mixed_precision'] else 'Disabled'}")
    
    try:
        # set verbose=0 to replace custom progress bar
        train_net.fit(train_ds, epochs=config['epoch'], steps_per_epoch=steps_per_epoch, verbose=0, callbacks=callbacks)
    except KeyboardInterrupt:
        print('--')
        if early_stop.best_weights is None:
            print('Training is canceled, but weights can not be restored because the best model is not available.')
        else:
            print('Training is canceled and weights are restored from the best')
            train_net.set_weights(early_stop.best_weights)

    # save current model at the end of training (final model)
    print("\n💾 Training completed - saving final model...")
    
    # create checkpoint directory
    checkpoint_dir = f"./checkpoints/{config['model_name']}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 1. save inference model (for TFLite conversion)
    infer_model = train_net.get_inference_model()
    final_inference_path = os.path.join(checkpoint_dir, 'final_inference.keras')
    infer_model.save(final_inference_path)
    print(f"   ✅ Final inference model: {final_inference_path}")
    
    # 2. save full model (for resume training) - remove learning rate scheduler
    final_full_path = os.path.join(checkpoint_dir, 'final_full.keras')
    # remove learning rate scheduler and save (for TFLite conversion)
    if train_net.optimizer is not None:
        temp_lr_schedule = getattr(train_net.optimizer, '_learning_rate', None)
        train_net.optimizer = None
        train_net.save(final_full_path)
        # do not restore learning rate scheduler (for TFLite conversion)
    else:
        train_net.save(final_full_path)
    print(f"   ✅ Final full model: {final_full_path}")
    
    # 3. save backbone model (for separate use)
    final_backbone_path = os.path.join(checkpoint_dir, 'final_backbone.keras')
    train_net.backbone.save(final_backbone_path)
    print(f"   ✅ Final backbone model: {final_backbone_path}")

    # 4. save final ckpt
    final_ckpt_dir = os.path.join(checkpoint_dir, 'ckpt')
    os.makedirs(final_ckpt_dir, exist_ok=True)
    final_ckpt = tf.train.Checkpoint(net=train_net)
    save_path = final_ckpt.save(os.path.join(final_ckpt_dir, 'ckpt'))
    print(f"   ✅ Final ckpt: {save_path}")
    
    print(f"\n📁 All models are saved: {checkpoint_dir}")
    print(f"   - best_inference.keras (best performance inference model)")
    print(f"   - best_full.keras (best performance full model)")
    print(f"   - best_full_backbone.keras (best performance backbone model)")
    print(f"   - final_inference.keras (final inference model)")
    print(f"   - final_full.keras (final full model)")
    print(f"   - final_backbone.keras (final backbone model)")
    print(f"   - training_log.csv (training log)")
    print(f"   - training_config.json (training config)")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FFEM Training')
    parser.add_argument('--config', type=str, default='train.config',
                        help='Config module path (default: train.config)')
    args = parser.parse_args()
    
    config_module = importlib.import_module(args.config)
    config = config_module.config
    
    print(f"📋 Using config: {args.config}")
    start_training(config)
