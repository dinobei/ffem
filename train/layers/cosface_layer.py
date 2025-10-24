import tensorflow as tf
from tensorflow.keras import layers
import math

class CosFaceLayer(layers.Layer):
    """
    CosFace (Additive Cosine Margin) Layer
    Paper: CosFace: Large Margin Cosine Loss for Deep Face Recognition
    """
    
    def __init__(self, num_classes, scale=30.0, margin=0.35, **kwargs):
        super(CosFaceLayer, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.scale = scale
        self.margin = margin
        
    def build(self, input_shape):
        """
        input_shape = [shape of y_pred, shape of y_true] : list
        """
        feature_dim = tf.TensorShape(input_shape[0]).as_list()[-1]
        self.W = self.add_weight(
            name='cosface_W',
            shape=(feature_dim, self.num_classes),
            initializer='glorot_uniform',
            trainable=True,
            dtype=tf.float32
        )
        
    def call(self, inputs):
        """
        inputs = [y_pred, y_true] : list
        y_pred = (batch_size, embedding dim) : shape
        y_true = (batch_size, classes) : shape
        """
        y_pred, y_true = inputs
        
        # Normalize inputs and weights
        x_norm = tf.nn.l2_normalize(y_pred, axis=1)
        w_norm = tf.nn.l2_normalize(self.W, axis=0)
        
        # Calculate cosine similarity
        cos_theta = tf.matmul(x_norm, w_norm)
        
        # Apply additive cosine margin
        cos_theta_m = cos_theta - self.margin
        cos_theta_m = tf.clip_by_value(cos_theta_m, -1.0, 1.0)
        
        # Scale the logits
        logits = tf.where(y_true == 1., cos_theta_m, cos_theta)
        logits = logits * self.scale
            
        return logits
    
    def get_config(self):
        config = super(CosFaceLayer, self).get_config()
        config.update({
            'num_classes': self.num_classes,
            'scale': self.scale,
            'margin': self.margin
        })
        return config

