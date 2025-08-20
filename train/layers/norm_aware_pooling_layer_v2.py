import tensorflow as tf


"""
An implementation of the paper:
Global Norm-Aware Pooling for Pose-Robust Face Recognition at Low False Positive Rate
Sheng Chen, Jia Guo, Yang Liu, Xiang Gao, Zhen Han
arXiv preprint arXiv:1808.00435 2018

Enhanced version with optional L2 normalization for GPU compatibility
"""
class NormAwarePoolingLayerV2(tf.keras.layers.Layer):

    def __init__(self, use_l2_norm=True):
        super(NormAwarePoolingLayerV2, self).__init__()
        self.use_l2_norm = use_l2_norm
        self.batchnorm_in = tf.keras.layers.BatchNormalization(scale=False)
        self.batchnorm_out = tf.keras.layers.BatchNormalization(scale=False)
        self.avg_pool = tf.keras.layers.GlobalAveragePooling2D()

    def call(self, x):
        if x.dtype == tf.float16:
            x = tf.cast(x, tf.float32)
        
        y = self.batchnorm_in(x)
        
        if self.use_l2_norm:
            # Original L2 normalization approach (GPU incompatible)
            norm = tf.norm(y, ord=2, axis=3, keepdims=True)
            norm = tf.clip_by_value(norm, 1e-8, 1e4)
            mean = tf.math.reduce_mean(norm)
            mean = tf.clip_by_value(mean, 1e-8, 1e4)
            y_norm = tf.math.l2_normalize(y, axis=3)
            y = tf.multiply(y_norm, mean)
        else:
            # GPU compatible approach: simple normalization without L2
            # Use mean and std normalization instead
            mean = tf.math.reduce_mean(y, axis=[1, 2, 3], keepdims=True)
            std = tf.math.reduce_std(y, axis=[1, 2, 3], keepdims=True)
            std = tf.clip_by_value(std, 1e-8, 1e4)
            y = (y - mean) / std
        
        y = self.avg_pool(y)
        y = self.batchnorm_out(y)
        
        if x.dtype == tf.float16:
            y = tf.cast(y, tf.float16)
        return y

    def get_config(self):
        config = super(NormAwarePoolingLayerV2, self).get_config()
        config.update({
            'use_l2_norm': self.use_l2_norm
        })
        return config

