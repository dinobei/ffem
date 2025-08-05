import tensorflow as tf


"""
An implementation of the paper:
Global Norm-Aware Pooling for Pose-Robust Face Recognition at Low False Positive Rate
Sheng Chen, Jia Guo, Yang Liu, Xiang Gao, Zhen Han
arXiv preprint arXiv:1808.00435 2018
"""
class NormAwarePoolingLayer(tf.keras.layers.Layer):

    def __init__(self):
        super(NormAwarePoolingLayer, self).__init__()
        self.batchnorm_in = tf.keras.layers.BatchNormalization(scale=False)
        self.batchnorm_out = tf.keras.layers.BatchNormalization(scale=False)
        self.avg_pool = tf.keras.layers.GlobalAveragePooling2D()

    def call(self, x):
        # 입력 차원 확인
        input_shape = tf.shape(x)
        input_rank = tf.rank(x)
        
        # 2차원 입력인 경우 (배치, 특성) - 단순히 반환
        if input_rank == 2:
            return x
        
        # 4차원 입력인 경우 (배치, 높이, 너비, 채널) - 원래 로직 적용
        elif input_rank == 4:
            y = self.batchnorm_in(x)
            norm = tf.norm(y, ord=2, axis=3, keepdims=True)
            mean = tf.math.reduce_mean(norm)
            y = tf.math.l2_normalize(y, axis=3)
            y = tf.multiply(y, mean)
            y = self.avg_pool(y)
            y = self.batchnorm_out(y)
            return y
        
        # 그 외의 경우 - 단순히 반환
        else:
            return x

    def get_config(self):
        return {}
