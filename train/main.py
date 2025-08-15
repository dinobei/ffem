import argparse
import importlib
import copy
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import train.input_pipeline as input_pipeline
import train.config
from train.callbacks import LogCallback
from train.callbacks import RecallCallback
from train.utils import apply_pruning
from train.utils import apply_quantization_aware
import net_arch.models
import train.blocks
from train.custom_models.softmax_center_model import SoftmaxCenterModel
from train.custom_models.angular_margin_model import AngularMarginModel
from train.custom_models.group_aware_model import GroupAwareModel

import tensorflow as tf
import numpy as np

def setup_multi_gpu():
    """멀티 GPU 설정"""
    gpus = tf.config.list_physical_devices('GPU')
    
    # 특정 GPU 선택 (환경변수에서 읽기)
    selected_gpus = os.environ.get('CUDA_VISIBLE_DEVICES', None)
    if selected_gpus:
        print(f"🎯 Using selected GPUs: {selected_gpus}")
        # 이미 환경변수로 설정되어 있으므로 현재 GPU 목록 사용
        available_gpus = tf.config.list_physical_devices('GPU')
        print(f"📱 Available GPUs: {len(available_gpus)}")
        
        if len(available_gpus) > 1:
            print(f"🚀 Found {len(available_gpus)} GPUs, enabling multi-GPU training")
            for gpu in available_gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            # GPU별 배치 크기 최적화
            strategy = create_optimized_strategy(available_gpus)
            return strategy
        else:
            print(f"📱 Using single GPU: {available_gpus[0] if available_gpus else 'CPU'}")
            return None
    else:
        # 기존 로직 (모든 GPU 사용)
        if len(gpus) > 1:
            print(f"🚀 Found {len(gpus)} GPUs, enabling multi-GPU training")
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            # GPU별 배치 크기 최적화
            strategy = create_optimized_strategy(gpus)
            return strategy
        else:
            print(f"📱 Using single GPU: {gpus[0] if gpus else 'CPU'}")
            return None

def create_optimized_strategy(gpus):
    """GPU 성능에 따른 최적화된 전략 생성"""
    # GPU 정보 수집
    gpu_info = []
    for i, gpu in enumerate(gpus):
        try:
            # GPU 메모리 정보 가져오기 (근사값)
            gpu_name = gpu.name
            if '3090' in gpu_name.lower():
                memory_gb = 24
                performance_score = 100
            elif '2080' in gpu_name.lower():
                memory_gb = 11
                performance_score = 60
            elif '3080' in gpu_name.lower():
                memory_gb = 10
                performance_score = 80
            else:
                memory_gb = 8  # 기본값
                performance_score = 50
            
            gpu_info.append({
                'index': i,
                'name': gpu_name,
                'memory_gb': memory_gb,
                'performance_score': performance_score
            })
        except:
            gpu_info.append({
                'index': i,
                'name': gpu.name,
                'memory_gb': 8,
                'performance_score': 50
            })
    
    print(f"📊 GPU Performance Analysis:")
    for info in gpu_info:
        print(f"  GPU {info['index']}: {info['name']} ({info['memory_gb']}GB, Score: {info['performance_score']})")
    
    # 성능 차이가 큰 경우 경고
    scores = [info['performance_score'] for info in gpu_info]
    if max(scores) - min(scores) > 30:
        print(f"⚠️  Large performance gap detected! Consider using only faster GPUs.")
        print(f"   Performance difference: {max(scores) - min(scores)} points")
    
    # MirroredStrategy 사용 (기본)
    strategy = tf.distribute.MirroredStrategy()
    print(f"✅ Using MirroredStrategy with {strategy.num_replicas_in_sync} replicas")
    
    return strategy

def get_optimized_batch_size(config, strategy):
    """GPU 성능에 따른 최적화된 배치 크기 계산"""
    if strategy is None:
        return config['batch_size']
    
    num_gpus = strategy.num_replicas_in_sync
    base_batch_size = config['batch_size']
    
    # GPU별 배치 크기 조정
    if num_gpus > 1:
        # 성능이 비슷한 GPU들: 배치 크기 증가
        # 성능 차이가 큰 GPU들: 배치 크기 조정
        adjusted_batch_size = base_batch_size * num_gpus
        
        print(f"📈 Batch size optimization:")
        print(f"  Base batch size: {base_batch_size}")
        print(f"  Adjusted batch size: {adjusted_batch_size} (per GPU: {adjusted_batch_size // num_gpus})")
        
        return adjusted_batch_size
    
    return base_batch_size

def setup_specific_gpu(gpu_indices):
    """특정 GPU만 사용하도록 설정"""
    if isinstance(gpu_indices, str):
        # 문자열로 받은 경우 (예: "0,1" 또는 "0")
        gpu_list = gpu_indices
    elif isinstance(gpu_indices, (list, tuple)):
        # 리스트나 튜플로 받은 경우 (예: [0, 1] 또는 [0])
        gpu_list = ','.join(map(str, gpu_indices))
    else:
        # 단일 정수로 받은 경우 (예: 0)
        gpu_list = str(gpu_indices)
    
    # 환경변수 설정
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
    print(f"🎯 Set CUDA_VISIBLE_DEVICES to: {gpu_list}")
    
    # GPU 설정 적용
    return setup_multi_gpu()


def count_dataset_samples(dataset_path):
    """데이터셋의 실제 샘플 수를 계산"""
    try:
        import tensorflow as tf
        import os
        from tqdm import tqdm
        
        print(f"🔍 Counting samples in: {dataset_path}")
        
        # 파일 크기로 빠른 추정 (대용량 데이터셋용)
        file_size = os.path.getsize(dataset_path)
        print(f"📁 File size: {file_size / (1024*1024):.1f} MB")
        
        # TFRecord 파일에서 실제 샘플 수 계산
        count = 0
        dataset = tf.data.TFRecordDataset(dataset_path)
        
        # tqdm으로 진행률바 표시
        with tqdm(desc="Counting samples", unit=" samples") as pbar:
            for record in dataset:
                count += 1
                pbar.update(1)
        
        print(f"📊 Actual dataset size: {count:,} samples")
        return count
        
    except Exception as e:
        print(f"⚠️  Could not count dataset samples: {e}")
        return None

def get_actual_dataset_size(config):
    """실제 데이터셋 크기 가져오기"""
    # 설정에서 가져오기 (None이면 자동 계산)
    if config.get('estimated_total_samples') is not None:
        total_samples = config['estimated_total_samples']
        samples_per_epoch = total_samples // config['epoch']
        print(f"📊 Using configured dataset size: {samples_per_epoch:,} samples per epoch")
        return samples_per_epoch
    
    # 실제 파일에서 계산
    train_file = config.get('train_file', '')
    if train_file and os.path.exists(train_file):
        actual_samples = count_dataset_samples(train_file)
        if actual_samples:
            return actual_samples
    
    # 기본값 (작은 테스트 데이터셋용)
    default_samples = 10000
    print(f"⚠️  Using default dataset size: {default_samples:,} samples per epoch")
    return default_samples


def build_dataset(config):
    train_ds, test_ds_dict = input_pipeline.make_tfdataset(
        config['train_file'],
        config['test_files'],
        config['batch_size'],
        config['shape'][:2])
    return train_ds, test_ds_dict


def build_backbone_model(config):
    is_pretrained = False
    if os.path.exists(config['saved_backbone']):
        net = tf.keras.models.load_model(config['saved_backbone'])
        print('\n---------------- Restore Backbone Network ----------------\n')
        print(config['saved_backbone'])
        print('\n----------------------------------------------------------\n')
        is_pretrained = True
    else :
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
    else:
        raise Exception('The loss ({}) is not supported.'.format(config['loss']))

    # Do pre-compile
    # This is a tensorflow bug.
    # After restoring a checkpoint without pre-compile, 
    #  learning rate is overridden with a checkpoint, when it compiles with a new optimizer.
    model.compile()
    restore_latest_checkpoint(model, config['checkpoint'])
    if config['enable_quant_aware']:
        model.backbone = apply_quantization_aware(model.backbone, None)
    model([dummy_x, dummy_y], training=True) # dummy call for building model
    return model


def build_callbacks(config, test_ds_dict):
    log_dir = os.path.join('logs', config['model_name'])
    callback_list = []
    metric = config['eval']['metric']
    recall_topk = config['eval']['recall']
    recall_eval = RecallCallback(test_ds_dict, recall_topk, metric, log_dir)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='recall@1', factor=0.5, mode='max',
        patience=2, min_lr=1e-4, verbose=1)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath='./checkpoints/{}/best.keras'.format(config['model_name']),
        monitor='recall@1',
        mode='max',
        save_best_only=True,
        verbose=1)
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='recall@1',
        mode='max', patience=7,
        restore_best_weights=True)
    tensorboard_log = LogCallback(log_dir)
    
    # 처리량 모니터링 콜백 추가
    from train.callbacks import ThroughputCallback, CustomProgressBar, NaNMonitorCallback
    
    # 실제 데이터셋 크기 가져오기
    samples_per_epoch = get_actual_dataset_size(config)
    total_samples = samples_per_epoch * config['epoch']
    
    throughput_monitor = ThroughputCallback(total_samples, log_dir)
    custom_progress = CustomProgressBar(samples_per_epoch, config['batch_size'])
    
    # NaN 모니터링 콜백 추가
    nan_monitor = NaNMonitorCallback(patience=10)

    callback_list.append(recall_eval)
    callback_list.append(checkpoint)
    callback_list.append(early_stop)
    if not config['lr_decay']:
        callback_list.append(reduce_lr)
    callback_list.append(tensorboard_log)
    callback_list.append(throughput_monitor)  # 처리량 모니터링 추가
    callback_list.append(custom_progress)  # 커스텀 진행률바 추가
    callback_list.append(nan_monitor)  # NaN 모니터링 추가
    return callback_list, early_stop


def build_optimizer(config):
    # In tf-v2.3.0, Do not use tf.keras.optimizers.schedules with ReduceLR callback.
    if config['lr_decay']:
        lr = tf.keras.optimizers.schedules.ExponentialDecay(
            config['lr'],
            decay_steps=config['lr_decay_steps'],
            decay_rate=config['lr_decay_rate'],
            staircase=True)
    else:
        lr = config['lr']

    opt_list = {
        'Adam': 
            tf.keras.optimizers.Adam(learning_rate=lr, clipnorm=0.5),
        'SGD':
            tf.keras.optimizers.SGD(learning_rate=lr,
                momentum=0.9, nesterov=True, clipnorm=0.5),
        'AdamW': 
            tf.keras.optimizers.AdamW(learning_rate=lr, weight_decay=1e-4, clipnorm=0.5),
    }
    if config['optimizer'] not in opt_list:
        print(config['optimizer'], 'is not support.')
        print('please select one of below.')
        print(opt_list.keys())
        exit(1)
    return opt_list[config['optimizer']]


def restore_latest_checkpoint(net, checkpoint_path):
    checkpoint = tf.train.Checkpoint(net)
    latest_path = tf.train.latest_checkpoint(checkpoint_path)
    print('\n---------------- Restore Checkpoint ----------------\n')
    if latest_path is not None:
        print('restore_latest_checkpoint:', latest_path)
        checkpoint.restore(latest_path).expect_partial()
    else:
        print('Can not find latest checkpoint file:', checkpoint_path)
    print('\n----------------------------------------------------\n')

def start_training(config):
    # Remove GPU timer warning
    os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
    os.environ['TF_GPU_THREAD_COUNT'] = '1'
    
    # 특정 GPU 선택
    if config.get('selected_gpus') is not None:
        strategy = setup_specific_gpu(config['selected_gpus'])
    else:
        strategy = setup_multi_gpu()
    
    if config['mixed_precision']:
        print('---------------- Enabled Mixed Precision ----------------')
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
    tf.get_logger().setLevel('ERROR')
    
    # 최적화된 배치 크기 계산
    optimized_batch_size = get_optimized_batch_size(config, strategy)
    
    # 배치 크기 업데이트된 설정 생성
    optimized_config = config.copy()
    optimized_config['batch_size'] = optimized_batch_size
        
    if strategy:
        with strategy.scope():
            train_ds, test_ds_dict = build_dataset(optimized_config)
            train_net = build_model(optimized_config)
            opt = build_optimizer(optimized_config)
            callbacks, early_stop = build_callbacks(optimized_config, test_ds_dict)
            train_net.compile(optimizer=opt)
    else:
        train_ds, test_ds_dict = build_dataset(optimized_config)
        train_net = build_model(optimized_config)
        opt = build_optimizer(optimized_config)
        callbacks, early_stop = build_callbacks(optimized_config, test_ds_dict)
        train_net.compile(optimizer=opt)
    
    train_net.summary()
    
    # steps_per_epoch 계산
    samples_per_epoch = get_actual_dataset_size(optimized_config)
    steps_per_epoch = samples_per_epoch // optimized_config['batch_size']
    print(f"🚀 Epoch 1/{config['epoch']} - Steps per epoch: {steps_per_epoch}")
    
    try:
        # verbose=0으로 설정하여 커스텀 진행률바가 출력을 대체
        train_net.fit(train_ds, epochs=config['epoch'], steps_per_epoch=steps_per_epoch, verbose=0, callbacks=callbacks)
    except KeyboardInterrupt:
        print('--')
        if early_stop.best_weights is None:
            print('Training is canceled, but weights can not be restored because the best model is not available.')
        else:
            print('Training is canceled and weights are restored from the best')
            train_net.set_weights(early_stop.best_weights)

    infer_model = train_net.get_inference_model()
    infer_model.save('{}.h5'.format(infer_model.name))
    train_net.backbone.save('{}_backbone.h5'.format(train_net.name))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FFEM Training')
    parser.add_argument('--config', type=str, default='train.config',
                        help='Config module path (default: train.config)')
    args = parser.parse_args()
    
    config_module = importlib.import_module(args.config)
    config = config_module.config
    
    print(f"📋 Using config: {args.config}")
    start_training(config)
