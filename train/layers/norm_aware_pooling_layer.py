import tensorflow as tf


"""
An implementation of the paper:
Global Norm-Aware Pooling for Pose-Robust Face Recognition at Low False Positive Rate
Sheng Chen, Jia Guo, Yang Liu, Xiang Gao, Zhen Han
arXiv preprint arXiv:1808.00435 2018
"""
@tf.keras.utils.register_keras_serializable()
class NormAwarePoolingLayer(tf.keras.layers.Layer):

    def __init__(self):
        super(NormAwarePoolingLayer, self).__init__()
        self.batchnorm_in = tf.keras.layers.BatchNormalization(scale=False)
        self.batchnorm_out = tf.keras.layers.BatchNormalization(scale=False)
        self.avg_pool = tf.keras.layers.GlobalAveragePooling2D()

    def call(self, x):
        if x.dtype == tf.float16:
            x = tf.cast(x, tf.float32)
        y = self.batchnorm_in(x)
        norm = tf.norm(y, ord=2, axis=3, keepdims=True)
        mean = tf.math.reduce_mean(norm)
        y_norm = tf.math.l2_normalize(y, axis=3)
        y = tf.multiply(y_norm, mean)
        y = self.avg_pool(y)
        y = self.batchnorm_out(y)
        if x.dtype == tf.float16:
            y = tf.cast(y, tf.float16)
        return y

    def get_config(self):
        return {}
