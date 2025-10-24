import os
import time
import math
import numpy as np
from tqdm import tqdm

import csv
import json
from datetime import datetime

import evaluate.recall as recall
from evaluate.clustering_metrics import calculate_all_clustering_metrics
from evaluate.embedding_space_analysis import calculate_all_embedding_metrics
from evaluate.accuracy_metrics import calculate_all_accuracy_metrics

import tensorflow as tf


class NaNMonitorCallback(tf.keras.callbacks.Callback):
    """detect NaN and early stop callback"""
    
    def __init__(self, patience=5):
        super(NaNMonitorCallback, self).__init__()
        self.patience = patience
        self.nan_count = 0
        self.last_valid_loss = None
        
    def on_train_batch_end(self, batch, logs=None):
        if logs is None:
            return
            
        current_loss = logs.get('loss', None)
        
        # detect NaN
        if current_loss is not None and (tf.math.is_nan(current_loss) or tf.math.is_inf(current_loss)):
            self.nan_count += 1
            print(f"\n⚠️  NaN/Inf detected! Batch {batch}, Loss: {current_loss}")
            print(f"   NaN count: {self.nan_count}/{self.patience}")
            
            if self.nan_count >= self.patience:
                print(f"\n❌ Training stopped due to NaN/Inf loss for {self.patience} consecutive batches")
                self.model.stop_training = True
        else:
            # reset counter if valid loss is found
            if current_loss is not None:
                self.nan_count = 0
                self.last_valid_loss = current_loss
                
        # warn if loss is too large
        if current_loss is not None and current_loss > 100:
            print(f"\n⚠️  High loss detected! Batch {batch}, Loss: {current_loss}")
            
    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            return
            
        epoch_loss = logs.get('loss', None)
        if epoch_loss is not None and (tf.math.is_nan(epoch_loss) or tf.math.is_inf(epoch_loss)):
            print(f"\n❌ Epoch {epoch + 1} ended with NaN/Inf loss: {epoch_loss}")
            self.model.stop_training = True


class CustomProgressBar(tf.keras.callbacks.Callback):
    """custom progress bar with throughput information"""
    
    def __init__(self, total_samples, batch_size, total_epochs=40):
        super(CustomProgressBar, self).__init__()
        self.epoch_start_time = None
        self.step_start_time = None
        self.last_batch = 0
        self.total_samples = total_samples
        self.batch_size = batch_size
        self.total_epochs = total_epochs
        self.steps_per_epoch = math.ceil(total_samples / batch_size)
        self.update_interval = max(1, self.steps_per_epoch // 1000)
        
        
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()
        self.step_start_time = time.time()
        self.last_batch = 0
        print(f"\n🚀 Epoch {epoch + 1}/{self.total_epochs} - Steps per epoch: {self.steps_per_epoch}")
        
    def on_train_batch_end(self, batch, logs=None):
        if logs is None:
            return
            
        # Extract throughput information
        throughput = logs.get('throughput', '')
        
        # Calculate current time
        current_time = time.time()
        step_time = current_time - self.step_start_time
        self.step_start_time = current_time
        
        # Extract performance information (for progress bar display)
        accuracy = logs.get('accuracy', 0.0)
        loss = logs.get('loss', 0.0)
        
        # Update progress bar with dynamically calculated update interval
        if batch % self.update_interval == 0:
            # Calculate epoch progress
            progress = (batch / self.steps_per_epoch) * 100
            
            # Simplified progress bar output
            progress_line = f"{batch}/{self.steps_per_epoch} ({progress:.1f}%) - acc: {accuracy:.4f} - loss: {loss:.4f}"
            if throughput:
                progress_line += f" {throughput}"
            
            print(f"\r{progress_line}", end='', flush=True)
            self.last_batch = batch
            
            # Newline at end of epoch (including last step)
            if batch >= self.steps_per_epoch - 1:
                final_line = f"{self.steps_per_epoch}/{self.steps_per_epoch} (100.0%) - acc: {accuracy:.4f} - loss: {loss:.4f}"
                if throughput:
                    final_line += f" {throughput}"
                print(f"\r{final_line}")
                print()  # Newline
    
    def on_epoch_end(self, epoch, logs=None):
        # Epoch end log is handled by RecallCallback
        pass


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
    """Throughput monitoring callback - combined progress bar"""
    
    def __init__(self, total_samples, log_dir='./logs', config=None):
        super(ThroughputCallback, self).__init__()
        self.total_samples = total_samples
        self.log_dir = log_dir
        self.writer = tf.summary.create_file_writer(log_dir)
        self.config = config
        
        # Performance measurement variables
        self.epoch_start_time = None
        self.step_samples = []
        
    def on_epoch_begin(self, epoch, logs=None):
        """Record epoch start time"""
        self.epoch_start_time = time.time()
        self.step_samples = []
        
    def on_train_batch_end(self, batch, logs=None):
        """Measure performance every batch end"""
        if self.epoch_start_time is not None and batch > 0:
            current_time = time.time()
            
            # Estimate batch size - use actual batch size
            # Get size from logs, if not available, use batch size from config
            default_batch_size = self.config.get('batch_size', 256) if self.config else 256
            batch_size = logs.get('size', default_batch_size) if logs else default_batch_size
            self.step_samples.append(batch_size)
            
            # Calculate real-time throughput
            elapsed_time = current_time - self.epoch_start_time
            total_samples_processed = sum(self.step_samples)
            samples_per_second = total_samples_processed / elapsed_time
            
            # Add throughput information to progress bar
            if logs is not None:
                logs['throughput'] = f"{samples_per_second:.1f} samples/sec"
    
    def on_epoch_end(self, epoch, logs=None):
        """Calculate final performance statistics at epoch end"""
        if self.epoch_start_time is not None and self.step_samples:
            total_time = time.time() - self.epoch_start_time
            total_samples = sum(self.step_samples)
            
            # Avoid ZeroDivisionError
            if total_samples > 0 and total_time > 0:
                # Calculate performance statistics
                samples_per_second = total_samples / total_time
                avg_time_per_sample = total_time / total_samples
            
                # GPU information
                gpu_count = len(tf.config.list_physical_devices('GPU'))
                # Effective batch size = actual batch size * GPU count * gradient accumulation
                num_grad_accum = self.config.get('num_grad_accum', 1) if self.config else 1
                effective_batch_size = self.step_samples[0] * gpu_count * num_grad_accum if self.step_samples else 0
                
                # Estimated remaining time for epochs
                remaining_epochs = 40 - (epoch + 1)  # get from config
                estimated_total_time = total_time * remaining_epochs
                estimated_days = estimated_total_time / (24 * 3600)
                
                print(f"\n📈 Epoch {epoch + 1} Performance Summary:")
                print(f"  ⏱️  Total time: {total_time:.2f} seconds ({total_time/3600:.2f} hours)")
                print(f"  📊 Total samples: {total_samples:,}")
                print(f"  🚀 Throughput: {samples_per_second:.1f} samples/second")
                print(f"  ⚡ Avg time per sample: {avg_time_per_sample*1000:.2f} ms")
                print(f"  🎯 Effective batch size: {effective_batch_size}")
                print(f"  🖥️  GPUs used: {gpu_count}")
                print(f"  ⏳ Estimated remaining time: {estimated_days:.1f} days")
                
                # Log to TensorBoard
                with self.writer.as_default():
                    tf.summary.scalar('throughput/samples_per_second', samples_per_second, step=epoch)
                    tf.summary.scalar('throughput/ms_per_sample', avg_time_per_sample * 1000, step=epoch)
                    tf.summary.scalar('throughput/effective_batch_size', effective_batch_size, step=epoch)
                    tf.summary.scalar('throughput/gpu_count', gpu_count, step=epoch)
                    tf.summary.scalar('time/epoch_time_hours', total_time/3600, step=epoch)
                    tf.summary.scalar('time/estimated_remaining_days', estimated_days, step=epoch)
                    self.writer.flush()
                
                # Add to logs
                logs['throughput_samples_per_sec'] = samples_per_second
                logs['throughput_ms_per_sample'] = avg_time_per_sample * 1000
                logs['effective_batch_size'] = effective_batch_size
                logs['gpu_count'] = gpu_count
                logs['epoch_time_hours'] = total_time/3600
                logs['estimated_remaining_days'] = estimated_days
            else:
                print(f"\n⚠️  Epoch {epoch + 1}: No samples processed or zero time elapsed")

    def on_train_end(self, logs=None):
        self.writer.close()


class WeightsCheckpoint(tf.keras.callbacks.Callback):
    """
    Weights-only checkpoint saving callback
    - Use tf.train.CheckpointManager to manage ckpt-* files
    - Save only when monitor metric improves (save_best_only=True)
    """

    def __init__(self, checkpoint_dir, monitor='val_loss', mode='min', save_best_only=True, verbose=1, max_to_keep=5):
        super(WeightsCheckpoint, self).__init__()
        self.checkpoint_dir = checkpoint_dir
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.verbose = verbose
        self.max_to_keep = max_to_keep

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            raise ValueError(f"mode {mode} is unknown")

        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self._ckpt = None
        self._manager = None

    def on_train_begin(self, logs=None):
        # Initialize after model binding
        self._ckpt = tf.train.Checkpoint(net=self.model)
        self._manager = tf.train.CheckpointManager(self._ckpt, self.checkpoint_dir, max_to_keep=self.max_to_keep)
        if self.verbose:
            print(f"📁 WeightsCheckpoint initialized: {self.checkpoint_dir}")

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current = logs.get(self.monitor)
        if current is None:
            if self.verbose:
                print(f"Warning: Monitor '{self.monitor}' not available. Skipping ckpt save.")
            return

        if self.save_best_only:
            if self.monitor_op(current, self.best):
                self.best = current
                self._manager.save()
                # Save log is handled by RecallCallback
        else:
            self._manager.save()


class RecallCallback(tf.keras.callbacks.Callback):
    """
    Extended Recall callback - includes advanced embedding quality metrics
    - Basic recall@k evaluation
    - Clustering quality metrics
    - Embedding space analysis
    - Accuracy metrics
    """

    def __init__(self, dataset_dict, top_k, metric, log_dir='logs', advanced_metrics_interval=5, model_name='unknown_model'):
        super(RecallCallback, self).__init__()
        self.ds_dict = dataset_dict
        self.top_k = top_k
        self.metric = metric
        self.log_dir = log_dir
        self.writer = tf.summary.create_file_writer(log_dir)
        self.advanced_metrics_interval = advanced_metrics_interval  # N epochs
        self.model_name = model_name
        
        # Create directory for advanced metrics
        self.advanced_log_dir = os.path.join(log_dir, 'advanced_metrics')
        os.makedirs(self.advanced_log_dir, exist_ok=True)

    def on_train_end(self, logs=None):
        self.writer.close()

    def on_epoch_end(self, epoch, logs=None):
        recall_avgs = {}
        # Init recall average dictionary
        for k in self.top_k:
            recall_avgs['recall@{}'.format(k)] = 0.
        # Evaluate recall over multiple datasets
        print(f"\n📊 Epoch {epoch + 1} - Analysis:")
        
        # Evaluate each dataset (internal tqdm display)
        dataset_names = list(self.ds_dict.keys())
        recall_results = {}
        
        for ds_name in dataset_names:
            ds = self.ds_dict[ds_name]
            ds_base_name = os.path.basename(ds_name)
            
            # Recall@K calculation (internal tqdm display)
            eval_start_time = time.time()
            recall_top_k = recall.evaluate(self.model, ds, self.metric, self.top_k, 256, dataset_name=ds_base_name)
            eval_time = time.time() - eval_start_time
            
            # Save results
            recall_results[ds_base_name] = {
                'recall_values': recall_top_k,
                'eval_time': eval_time
            }
            
            # Basic recall metrics logging
            with self.writer.as_default():
                for k, value in zip(self.top_k, recall_top_k):
                    recall_str = 'recall@{}'.format(k)
                    scalar_name = ds_base_name + '_{}'.format(recall_str)
                    value *= 100
                    tf.summary.scalar(scalar_name, value, step=epoch)
                    recall_avgs[recall_str] += value
                self.writer.flush()
            
            # Embedding quality analysis (only for first dataset, N epochs)
            # if ds_name == dataset_names[0]:  # first dataset
            #     if (epoch + 1) % self.advanced_metrics_interval == 0 or epoch == 0:
            #         self._evaluate_advanced_metrics(epoch, ds, ds_base_name, logs)
        
        # Print results after evaluating all datasets
        print("\n📋 Evaluation Results:")
        for ds_base_name, result in recall_results.items():
            recall_values = [f"{value*100:.2f}" for value in result['recall_values']]
            recall_k_str = ",".join([str(k) for k in self.top_k])
            recall_values_str = ",".join(recall_values)
            print(f"  🔍 {ds_base_name}")
            print(f"    Recall@{recall_k_str}: {recall_values_str} ({result['eval_time']:.1f}s)")
        
        # Save individual dataset results to logs
        for ds_base_name, result in recall_results.items():
            for k, value in zip(self.top_k, result['recall_values']):
                logs[f'{ds_base_name}_recall@{k}'] = value * 100
        
        # Calculate overall recall average
        with self.writer.as_default():
            ds_size = len(self.ds_dict)
            for key in recall_avgs:
                recall_avgs[key] /= ds_size
                logs[key] = recall_avgs[key]
            self.writer.flush()
        
        # recall@1 improved, print combined save log
        if 'recall@1' in recall_avgs:
            current_recall = recall_avgs['recall@1']
            if not hasattr(self, 'best_recall'):
                self.best_recall = -np.Inf
            
            if current_recall > self.best_recall:
                print(f"\nEpoch {epoch + 1}: recall@1 improved from {self.best_recall:.5f} to {current_recall:.5f}")
                # Get model name dynamically from callback list
                model_name = getattr(self, 'model_name', 'unknown_model')
                checkpoint_dir = f"./checkpoints/{model_name}"
                print(f"saving inference model to {checkpoint_dir}/best_inference.keras")
                print(f"saving backbone model to {checkpoint_dir}/best_full_backbone.keras")
                print(f"saving full model to {checkpoint_dir}/best_full.keras")
                print(f"saving ckpt to {checkpoint_dir}/ckpt")
                self.best_recall = current_recall
        
        # Print combined epoch end log
        print(f"\n✅ Epoch {epoch + 1} completed!")

    def _evaluate_advanced_metrics(self, epoch, dataset, dataset_name, logs):
        """Calculate embedding quality metrics"""
        try:
            # Extract embeddings
            embeddings, labels = self._extract_embeddings(dataset)
            
            # 1. Clustering quality metrics
            clustering_metrics = calculate_all_clustering_metrics(embeddings, labels)
            
            # 2. Embedding space analysis
            embedding_metrics = calculate_all_embedding_metrics(embeddings, labels)
            
            # 3. Accuracy metrics
            accuracy_metrics = calculate_all_accuracy_metrics(embeddings, labels)
            
            # Add all metrics to logs
            all_metrics = {**clustering_metrics, **embedding_metrics, **accuracy_metrics}
            
            # Log to TensorBoard
            with self.writer.as_default():
                for key, value in all_metrics.items():
                    if isinstance(value, (int, float)) and not np.isnan(value):
                        tf.summary.scalar(f'advanced_metrics/{key}', value, step=epoch)
                        logs[f'advanced_{key}'] = value
                self.writer.flush()
            
            # Print main metrics cleanly
            self._print_clean_metrics(all_metrics)
            
        except Exception as e:
            print(f"    ❌ Error: {e}")

    def _extract_embeddings(self, dataset):
        """Extract embeddings from dataset (optimized batch size)"""
        X = []
        Y = []
        
        # Use maximum 5000 samples (computational efficiency)
        max_samples = 5000
        samples_collected = 0
        
        for batch_x, batch_y in dataset:
            if samples_collected >= max_samples:
                break
                
            batch_pred = self.model(batch_x)
            # L2 normalization
            batch_pred = tf.math.l2_normalize(batch_pred, axis=1)
            
            # Add batch size, but limit to maximum samples
            batch_size = batch_pred.shape[0]
            remaining_samples = max_samples - samples_collected
            
            if batch_size <= remaining_samples:
                X.append(batch_pred.numpy())
                Y.append(batch_y.numpy())
                samples_collected += batch_size
            else:
                # Use only as many samples as needed in the last batch
                X.append(batch_pred.numpy()[:remaining_samples])
                Y.append(batch_y.numpy()[:remaining_samples])
                samples_collected += remaining_samples
                break
        
        embeddings = np.concatenate(X, axis=0)
        labels = np.concatenate(Y, axis=0)
        
        return embeddings, labels

    def _print_clean_metrics(self, metrics):
        """Print main metrics cleanly"""
        # Organize by category
        print(f"    Spread: {metrics.get('mean_spread', 0):.4f}")
        print(f"    Consistency: {metrics.get('mean_consistency', 0):.4f}")
        print(f"    Distribution: {metrics.get('norm_consistency', 0):.4f}")
        
        # Clustering quality
        if metrics.get('silhouette_score') is not None:
            print(f"    Silhouette: {metrics['silhouette_score']:.3f}")
        if metrics.get('davies_bouldin_index') is not None:
            print(f"    DBI: {metrics['davies_bouldin_index']:.3f}")
        
        # Distance analysis
        if 'margin' in metrics:
            print(f"    Margin: {metrics['margin']:.3f}")
        if 'margin_ratio' in metrics:
            print(f"    Ratio: {metrics['margin_ratio']:.1f}")
        
        # Accuracy
        if 'map' in metrics:
            print(f"    MAP: {metrics['map']:.3f}")
        
        # Precision@K (K=1,5 only)
        for k in [1, 5]:
            if f'precision@{k}' in metrics:
                print(f"    P@{k}: {metrics[f'precision@{k}']:.3f}")


class InferenceModelCheckpoint(tf.keras.callbacks.Callback):
    """Inference model only checkpoint callback"""
    
    def __init__(self, filepath, monitor='val_loss', mode='min', save_best_only=True, verbose=1):
        super(InferenceModelCheckpoint, self).__init__()
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.verbose = verbose
        
        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            raise ValueError(f"mode {mode} is unknown")
    
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current = logs.get(self.monitor)
        
        if current is None:
            if self.verbose > 0:
                print(f"\nWarning: Monitor '{self.monitor}' is not available. Skipping checkpoint save.")
            return
        
        if self.save_best_only:
            if self.monitor_op(current, self.best):
                self.best = current
                self._save_inference_model()
                # Save log is handled by RecallCallback
        else:
            self._save_inference_model()
    
    def _save_inference_model(self):
        """Save inference model"""
        try:
            # Create directory if it doesn't exist
            import os
            save_dir = os.path.dirname(self.filepath)
            if save_dir and not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)
                if self.verbose > 0:
                    print(f"📁 Created directory: {save_dir}")
            
            # Extract inference model
            if hasattr(self.model, 'get_inference_model'):
                inference_model = self.model.get_inference_model()
                # remove learning rate scheduler and save (for TFLite conversion)
                inference_model.save(self.filepath)
            else:
                # Save full model (fallback) - remove learning rate scheduler
                temp_optimizer = getattr(self.model, 'optimizer', None)
                if temp_optimizer is not None:
                    self.model.optimizer = None
                self.model.save(self.filepath)
                if temp_optimizer is not None:
                    self.model.optimizer = temp_optimizer  # restore
        except Exception as e:
            print(f"Error saving inference model: {e}")
            # Try to save full model on error
            try:
                self.model.save(self.filepath)
            except Exception as e2:
                print(f"Fallback save also failed: {e2}")


class CSVLoggerCallback(tf.keras.callbacks.Callback):
    """
    Callback to save performance metrics to CSV file
    - Recall@K metrics
    - Precision@K metrics (optional)
    - Training settings information
    - Timestamp
    """
    
    def __init__(self, log_dir, config, test_ds_dict=None, top_k=[1, 5, 10], 
                 metric='cosine', include_precision=True, save_config=True):
        super(CSVLoggerCallback, self).__init__()
        self.log_dir = log_dir
        self.config = config
        self.test_ds_dict = test_ds_dict
        self.top_k = top_k
        self.metric = metric
        self.include_precision = include_precision
        self.save_config = save_config
        
        # CSV file path setting
        self.csv_file = os.path.join(log_dir, 'training_log.csv')
        self.config_file = os.path.join(log_dir, 'training_config.json')
        
        # CSV header initialization
        self._init_csv_file()
        
        # Save settings file
        if self.save_config:
            self._save_config()
    
    def _init_csv_file(self):
        """CSV file header initialization"""
        os.makedirs(self.log_dir, exist_ok=True)
        
        # CSV header definition
        headers = [
            'epoch', 'timestamp', 'elapsed_time',
            'train_loss', 'train_accuracy', 'learning_rate'
        ]
        
        # Recall@K header addition
        for k in self.top_k:
            headers.append(f'recall@{k}')
        
        # Precision@K header addition (optional)
        if self.include_precision:
            for k in self.top_k:
                headers.append(f'precision@{k}')
        
        # Dataset-specific header addition
        if self.test_ds_dict:
            for ds_name in self.test_ds_dict.keys():
                ds_base_name = os.path.basename(ds_name)
                for k in self.top_k:
                    headers.append(f'{ds_base_name}_recall@{k}')
                if self.include_precision:
                    for k in self.top_k:
                        headers.append(f'{ds_base_name}_precision@{k}')
        
        # CSV file creation or header check
        file_exists = os.path.exists(self.csv_file)
        with open(self.csv_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(headers)
        
        self.headers = headers
        print(f"📊 CSV logging initialized: {self.csv_file}")
    
    def _save_config(self):
        """Save training settings to JSON file"""
        config_to_save = {
            'model_name': self.config.get('model_name', 'unknown'),
            'model': self.config.get('model', 'unknown'),
            'embedding_dim': self.config.get('embedding_dim', 512),
            'batch_size': self.config.get('batch_size', 128),
            'num_grad_accum': self.config.get('num_grad_accum', 1),
            'effective_batch_size': self.config.get('batch_size', 128) * self.config.get('num_grad_accum', 1),
            'learning_rate': self.config.get('lr', 0.001),
            'optimizer': self.config.get('optimizer', 'adam'),
            'loss': self.config.get('loss', 'cosface'),
            'epochs': self.config.get('epoch', 100),
            'shape': self.config.get('shape', [112, 112, 3]),
            'num_identity': self.config.get('num_identity', 5746),
            'mixed_precision': self.config.get('mixed_precision', False),
            'loss_param': self.config.get('loss_param', {}),
            'train_file': self.config.get('train_file', ''),
            'test_files': self.config.get('test_files', []),
            'top_k': self.top_k,
            'metric': self.metric,
            'include_precision': self.include_precision,
            'created_at': datetime.now().isoformat(),
            'log_dir': self.log_dir
        }
        
        with open(self.config_file, 'w', encoding='utf-8') as f:
            json.dump(config_to_save, f, indent=2, ensure_ascii=False)
        
        print(f"📋 Settings file saved: {self.config_file}")
    
    def on_train_begin(self, logs=None):
        """Called when training begins"""
        self.start_time = time.time()
        self.epoch_times = []
        print(f"🚀 Training started - CSV logging enabled")
        print(f"   Log file: {self.csv_file}")
        print(f"   Settings file: {self.config_file}")
    
    def on_epoch_end(self, epoch, logs=None):
        """Called when epoch ends"""
        if logs is None:
            logs = {}
        
        # Current time
        current_time = datetime.now()
        timestamp = current_time.strftime('%Y-%m-%d %H:%M:%S')
        
        # Calculate elapsed time
        elapsed_time = time.time() - self.start_time
        
        # Calculate epoch-wise elapsed time
        if hasattr(self, 'epoch_start_time'):
            epoch_time = time.time() - self.epoch_start_time
            self.epoch_times.append(epoch_time)
        else:
            epoch_time = 0
        
        # Collect basic metrics
        row_data = {
            'epoch': epoch + 1,
            'timestamp': timestamp,
            'elapsed_time': round(elapsed_time, 2),
            'train_loss': logs.get('loss', 0.0),
            'train_accuracy': logs.get('accuracy', 0.0),
            'learning_rate': float(self.model.optimizer.learning_rate.numpy()) if hasattr(self.model.optimizer, 'learning_rate') else 0.0
        }
        
        # Recall@K and Precision@K are calculated by RecallCallback, so get from logs
        for k in self.top_k:
            row_data[f'recall@{k}'] = logs.get(f'recall@{k}', 0.0)
            if self.include_precision:
                row_data[f'precision@{k}'] = logs.get(f'precision@{k}', 0.0)
        
        # Recall@K of individual datasets are also get from logs
        if self.test_ds_dict:
            for ds_name in self.test_ds_dict.keys():
                ds_base_name = os.path.basename(ds_name)
                for k in self.top_k:
                    row_data[f'{ds_base_name}_recall@{k}'] = logs.get(f'{ds_base_name}_recall@{k}', 0.0)
                    if self.include_precision:
                        row_data[f'{ds_base_name}_precision@{k}'] = logs.get(f'{ds_base_name}_precision@{k}', 0.0)
        
        # Write to CSV file
        self._write_to_csv(row_data)
        
        # Record next epoch start time
        self.epoch_start_time = time.time()
        
        # Progress output is handled by RecallCallback
    
    def on_epoch_begin(self, epoch, logs=None):
        """Called when epoch begins"""
        self.epoch_start_time = time.time()
    
    
    def _write_to_csv(self, row_data):
        """Write data to CSV file"""
        with open(self.csv_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            row = []
            for header in self.headers:
                row.append(row_data.get(header, ''))
            writer.writerow(row)
    
    def on_train_end(self, logs=None):
        """Called when training ends"""
        total_time = time.time() - self.start_time
        print(f"✅ Training completed - Total time: {total_time/3600:.2f} hours")
        print(f"📊 Log file: {self.csv_file}")
        
        # Save training summary information
        summary_file = os.path.join(self.log_dir, 'training_summary.json')
        summary = {
            'total_training_time': total_time,
            'total_epochs': len(self.epoch_times),
            'average_epoch_time': np.mean(self.epoch_times) if self.epoch_times else 0,
            'csv_file': self.csv_file,
            'config_file': self.config_file,
            'completed_at': datetime.now().isoformat()
        }
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"📋 Training summary: {summary_file}")


class BackboneCheckpoint(tf.keras.callbacks.Callback):
    """
    Callback to save backbone model also
    """
    
    def __init__(self, filepath, monitor='val_loss', mode='min', 
                 save_best_only=True, verbose=0, save_weights_only=False):
        super(BackboneCheckpoint, self).__init__()
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.verbose = verbose
        self.save_weights_only = save_weights_only
        
        # Determine direction of metric to monitor
        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            raise ValueError(f"Mode {mode} is unknown, use 'min' or 'max'")
        
        # Backbone model save path
        self.backbone_filepath = filepath.replace('.keras', '_backbone.keras')
        
    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        
        current = logs.get(self.monitor)
        if current is None:
            if self.verbose > 0:
                print(f"\nEpoch {epoch + 1}: {self.monitor} not available, skipping model save")
            return
        
        # Check if current performance is better
        if self.save_best_only:
            if self.monitor_op(current, self.best):
                self.best = current
                self._save_models(epoch, logs)
                # Save log is handled by RecallCallback
        else:
            self._save_models(epoch, logs)
    
    def _save_models(self, epoch, logs):
        """Save models"""
        try:
            # Create directory
            os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
            
            # Save full model - remove learning rate scheduler and save
            if self.save_weights_only:
                self.model.save_weights(self.filepath)
            else:
                temp_optimizer = getattr(self.model, 'optimizer', None)
                if temp_optimizer is not None:
                    self.model.optimizer = None
                self.model.save(self.filepath)
                if temp_optimizer is not None:
                    self.model.optimizer = temp_optimizer  # restore

            # Save backbone model - remove learning rate scheduler and save
            if hasattr(self.model, 'backbone') and self.model.backbone is not None:
                if self.save_weights_only:
                    self.model.backbone.save_weights(self.backbone_filepath)
                else:
                    self.model.backbone.save(self.backbone_filepath)
                
                # Save log is handled by RecallCallback
            
            # Save log is handled by RecallCallback
                
        except Exception as e:
            print(f"❌ Error saving models: {e}")
