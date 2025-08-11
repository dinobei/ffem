from train.utils import pairwise_distance

import tensorflow as tf


"""
An implementation of the paper:
A Discriminative Feature Learning Approach for Deep Face Recognition
Yandong Wen, Kaipeng Zhang, Zhifeng Li, and Yu Qiao
ECCV2016
"""
class CenterMarginLayer(tf.keras.layers.Layer):

    def __init__(self, num_classes, scale=30):
        super(CenterMarginLayer, self).__init__()
        self.num_classes = num_classes
        self.scale = scale

    def build(self, input_shape):
        n_embeddings = tf.TensorShape(input_shape[0]).as_list()[-1]
        initializer = tf.keras.initializers.HeNormal()
        self.c = self.add_weight(
            name='centers',
            shape=(self.num_classes, n_embeddings),
            initializer=initializer,
            trainable=True
        )

    def call(self, inputs):
        y_pred, y_true = inputs
        current_dtype = y_pred.dtype
        c = tf.cast(self.c, current_dtype)
        
        if current_dtype == tf.float16:
            y_pred = tf.cast(y_pred, tf.float32)
            c = tf.cast(c, tf.float32)
        
        y_pred_norm = tf.clip_by_value(y_pred, -1e6, 1e6)
        c_norm = tf.clip_by_value(c, -1e6, 1e6)
        
        norm_x = tf.math.l2_normalize(y_pred_norm, axis=1)
        norm_c = tf.math.l2_normalize(c_norm, axis=1)
        
        dist = pairwise_distance(norm_x, norm_c) * self.scale
        loss = tf.where(y_true == 1., dist, tf.zeros_like(dist))
        loss = tf.math.reduce_sum(loss, axis=1)
        
        if current_dtype == tf.float16:
            loss = tf.cast(loss, current_dtype)
        
        return loss

    def get_config(self):
        return {'num_classes': self.num_classes, 'scale': self.scale}
