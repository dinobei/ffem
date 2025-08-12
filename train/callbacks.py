import os
import time

import evaluate.recall as recall

import tensorflow as tf


class CustomProgressBar(tf.keras.callbacks.Callback):
    """처리량 정보를 포함한 커스텀 진행률바"""
    
    def __init__(self):
        super(CustomProgressBar, self).__init__()
        self.epoch_start_time = None
        self.step_start_time = None
        self.last_batch = 0
        
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()
        self.step_start_time = time.time()
        self.last_batch = 0
        
    def on_train_batch_end(self, batch, logs=None):
        if logs is None:
            return
            
        # 처리량 정보 추출
        throughput = logs.get('throughput', '')
        
        # 현재 시간 계산
        current_time = time.time()
        step_time = current_time - self.step_start_time
        self.step_start_time = current_time
        
        # 진행률바 업데이트 (기존 출력 대체)
        if batch % 10 == 0:  # 10배치마다 업데이트
            # 기존 진행률바 형식에 처리량 추가
            accuracy = logs.get('accuracy', 0.0)
            loss = logs.get('loss', 0.0)
            
            # 진행률바 출력 (기존 형식 + 처리량)
            progress_line = f"{batch}/Unknown {step_time:.0f}s {step_time:.1f}s/step - accuracy: {accuracy:.3e} - loss: {loss:.4f}"
            if throughput:
                progress_line += f" {throughput}"
            
            print(f"\r{progress_line}", end='', flush=True)
            self.last_batch = batch
    
    def on_epoch_end(self, epoch, logs=None):
        print()  # 줄바꿈


class LogCallback(tf.keras.callbacks.Callback):

    def __init__(self, log_dir='./logs'):
        super(LogCallback, self).__init__()
        self.log_dir = log_dir
        self.writer = tf.summary.create_file_writer(log_dir)

    def on_train_end(self, logs=None):
        self.writer.close()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        with self.writer.as_default():
            for name, value in logs.items():
                tf.summary.scalar(name, value, step=epoch)
            self.writer.flush()


class ThroughputCallback(tf.keras.callbacks.Callback):
    """처리량(throughput) 모니터링 콜백 - 진행률바 통합"""
    
    def __init__(self, total_samples, log_dir='./logs'):
        super(ThroughputCallback, self).__init__()
        self.total_samples = total_samples
        self.log_dir = log_dir
        self.writer = tf.summary.create_file_writer(log_dir)
        
        # 성능 측정 변수
        self.epoch_start_time = None
        self.step_samples = []
        
    def on_epoch_begin(self, epoch, logs=None):
        """에포크 시작 시 시간 기록"""
        self.epoch_start_time = time.time()
        self.step_samples = []
        
    def on_train_batch_end(self, batch, logs=None):
        """배치 끝날 때마다 성능 측정"""
        if self.epoch_start_time is not None and batch > 0:
            current_time = time.time()
            
            # 배치 크기 추정
            batch_size = logs.get('size', 32) if logs else 32
            self.step_samples.append(batch_size)
            
            # 실시간 처리량 계산
            elapsed_time = current_time - self.epoch_start_time
            total_samples_processed = sum(self.step_samples)
            samples_per_second = total_samples_processed / elapsed_time
            
            # 진행률바에 처리량 정보 추가
            if logs is not None:
                logs['throughput'] = f"{samples_per_second:.1f} samples/sec"
    
    def on_epoch_end(self, epoch, logs=None):
        """에포크 끝날 때 최종 성능 통계"""
        if self.epoch_start_time is not None:
            total_time = time.time() - self.epoch_start_time
            total_samples = sum(self.step_samples)
            
            # 성능 통계 계산
            samples_per_second = total_samples / total_time
            avg_time_per_sample = total_time / total_samples
            
            # GPU 정보
            gpu_count = len(tf.config.list_physical_devices('GPU'))
            effective_batch_size = self.step_samples[0] * gpu_count if self.step_samples else 0
            
            print(f"\n📈 Epoch {epoch + 1} Performance Summary:")
            print(f"  ⏱️  Total time: {total_time:.2f} seconds")
            print(f"  📊 Total samples: {total_samples:,}")
            print(f"  🚀 Throughput: {samples_per_second:.1f} samples/second")
            print(f"  ⚡ Avg time per sample: {avg_time_per_sample*1000:.2f} ms")
            print(f"  🎯 Effective batch size: {effective_batch_size}")
            print(f"  🖥️  GPUs used: {gpu_count}")
            
            # TensorBoard에 기록
            with self.writer.as_default():
                tf.summary.scalar('throughput/samples_per_second', samples_per_second, step=epoch)
                tf.summary.scalar('throughput/ms_per_sample', avg_time_per_sample * 1000, step=epoch)
                tf.summary.scalar('throughput/effective_batch_size', effective_batch_size, step=epoch)
                tf.summary.scalar('throughput/gpu_count', gpu_count, step=epoch)
                self.writer.flush()
            
            # 로그에 추가
            logs['throughput_samples_per_sec'] = samples_per_second
            logs['throughput_ms_per_sample'] = avg_time_per_sample * 1000
            logs['effective_batch_size'] = effective_batch_size
            logs['gpu_count'] = gpu_count

    def on_train_end(self, logs=None):
        self.writer.close()


class RecallCallback(tf.keras.callbacks.Callback):

    def __init__(self, dataset_dict, top_k, metric, log_dir='logs'):
        super(RecallCallback, self).__init__()
        self.ds_dict = dataset_dict
        self.top_k = top_k
        self.metric = metric
        self.log_dir = log_dir
        self.writer = tf.summary.create_file_writer(log_dir)

    def on_train_end(self, logs=None):
        self.writer.close()

    def on_epoch_end(self, epoch, logs=None):
        recall_avgs = {}
        # Init recall average dictionary
        for k in self.top_k:
            recall_avgs['recall@{}'.format(k)] = 0.
        # Evaluate recall over multiple datasets
        for ds_name in self.ds_dict:
            ds = self.ds_dict[ds_name]
            ds_base_name = os.path.basename(ds_name)
            recall_top_k = recall.evaluate(self.model, ds, self.metric, self.top_k, 256)
            with self.writer.as_default():
                for k, value in zip(self.top_k, recall_top_k):
                    recall_str = 'recall@{}'.format(k)
                    scalar_name = ds_base_name + '_{}'.format(recall_str)
                    value *= 100
                    tf.summary.scalar(scalar_name, value, step=epoch)
                    logs[recall_str] = tf.identity(value)
                    recall_avgs[recall_str] += value
                self.writer.flush()
        with self.writer.as_default():
            ds_size = len(self.ds_dict)
            for key in recall_avgs:
                recall_avgs[key] /= ds_size
                logs[key] = recall_avgs[key]
            self.writer.flush()
