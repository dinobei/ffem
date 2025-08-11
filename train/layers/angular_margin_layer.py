import math
import tensorflow as tf

"""
An implementation of the paper:
ArcFace: Additive Angular Margin Loss for Deep Face Recognition
Jiankang Deng, Jia Guo, Niannan Xue, Stefanos Zafeiriou
CVPR2019

All the code below is referenced from :
https://github.com/peteryuX/arcface-tf2
"""

class AngularMarginLayer(tf.keras.layers.Layer):

    def __init__(self, n_classes, margin=0.5, scale=30):
        super(AngularMarginLayer, self).__init__()
        self.n_classes = n_classes
        self.scale = scale
        self.margin = margin
        self.initializer = tf.keras.initializers.HeNormal()
        self.cos_m = tf.constant(math.cos(margin))
        self.sin_m = tf.constant(math.sin(margin))
        self.th = tf.constant(math.cos(math.pi - margin))
        self.mm = tf.constant(self.sin_m * margin)

    def build(self, input_shape):
        """
        input_shape = [shape of y_pred, shape of y_true] : list
        """
        feature_dim = tf.TensorShape(input_shape[0]).as_list()[-1]
        self.center = self.add_weight(
            name='center',
            shape=(feature_dim, self.n_classes),
            initializer=self.initializer,
            trainable=True
        )

    def call(self, inputs):
        """
        inputs = [y_pred, y_true] : list
        y_pred = (batch_size, embedding dim) : shape
        y_true = (batch_size, classes) : shape
        """
        y_pred, y_true = inputs
        
        current_dtype = y_pred.dtype
        center = tf.cast(self.center, current_dtype)
        
        cos_m = tf.cast(self.cos_m, current_dtype)
        sin_m = tf.cast(self.sin_m, current_dtype)
        th = tf.cast(self.th, current_dtype)
        mm = tf.cast(self.mm, current_dtype)
        
        if current_dtype == tf.float16:
            y_pred = tf.cast(y_pred, tf.float32)
            center = tf.cast(center, tf.float32)
            cos_m = tf.cast(cos_m, tf.float32)
            sin_m = tf.cast(sin_m, tf.float32)
            th = tf.cast(th, tf.float32)
            mm = tf.cast(mm, tf.float32)
        
        y_pred_norm = tf.clip_by_value(y_pred, -1e6, 1e6)
        center_norm = tf.clip_by_value(center, -1e6, 1e6)
        
        normed_embds = tf.nn.l2_normalize(y_pred_norm, axis=1)
        normed_w = tf.nn.l2_normalize(center_norm, axis=0)
        
        cos_t = tf.matmul(normed_embds, normed_w)
        
        cos_t = tf.clip_by_value(cos_t, -0.9999, 0.9999)
        
        sin_t_squared = 1.0 - cos_t * cos_t
        sin_t_squared = tf.clip_by_value(sin_t_squared, 1e-7, 1.0)
        sin_t = tf.sqrt(sin_t_squared)
        
        cos_mt = cos_t * cos_m - sin_t * sin_m
        cos_mt = tf.where(cos_t > th, cos_mt, cos_t - mm)
        
        logits = tf.where(y_true == 1., cos_mt, cos_t)
        logits = logits * self.scale
        
        if current_dtype == tf.float16:
            logits = tf.cast(logits, current_dtype)
        
        return logits

    def get_config(self):
        return {"n_classes": self.n_classes,
            'margin': self.margin,
            'scale': self.scale}
