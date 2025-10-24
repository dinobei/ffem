import tensorflow as tf
from train.utils import GradientAccumulatorModel
from train.layers.adaface_layer import AdaFaceLayer
from train.layers.norm_aware_pooling_layer import NormAwarePoolingLayer

class AdaFaceModel(GradientAccumulatorModel):
    def __init__(self,
                 backbone,
                 n_classes,
                 embedding_dim=512,
                 scale=30.0,
                 margin=0.4,
                 h=0.333,
                 num_grad_accum=1,
                 dropout_rate=0.1,
                 **kargs):
        super(AdaFaceModel, self).__init__(num_accum=num_grad_accum, **kargs)
        self.backbone = backbone
        self.n_classes = n_classes
        
        # use NormAwarePoolingLayer
        self.feature_pooling = NormAwarePoolingLayer()
        self.fc1 = tf.keras.layers.Dense(embedding_dim,
            kernel_regularizer=tf.keras.regularizers.l2(5e-4))
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.batchnorm_final = tf.keras.layers.BatchNormalization()
        self.adaface_layer = AdaFaceLayer(n_classes, scale, margin, h)
        self.loss_tracker = tf.keras.metrics.Mean(name='loss')
        self.acc_tracker = tf.keras.metrics.CategoricalAccuracy()

    def compile(self, **kargs):
        super(AdaFaceModel, self).compile(**kargs)

    def call(self, inputs, training=False):
        if training:
            x, y_true = inputs
            features = self.backbone(x)
            embeddings = self.feature_pooling(features)
            embeddings = self.fc1(embeddings)
            embeddings = self.dropout(embeddings, training=True)
            embeddings = self.batchnorm_final(embeddings)
            embeddings = self.adaface_layer([embeddings, y_true])
        else:
            x = inputs
            features = self.backbone(x)
            embeddings = self.feature_pooling(features)
            embeddings = self.fc1(embeddings)
            embeddings = self.dropout(embeddings, training=False)
            embeddings = self.batchnorm_final(embeddings)
        return embeddings

    def train_step(self, data):
        x, y_true = data
        with tf.GradientTape() as tape:
            y_true = tf.one_hot(y_true, self.n_classes)
            y_pred = self([x, y_true], training=True)
            total_loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred, from_logits=True)
            total_loss = tf.math.reduce_mean(total_loss)
        grads = tape.gradient(total_loss, self.trainable_variables)
        self.accumulate_grads_and_apply(grads)
        self.loss_tracker.update_state(total_loss)
        probs = tf.nn.softmax(y_pred, axis=1)
        self.acc_tracker.update_state(y_true, probs)
        return {'loss': self.loss_tracker.result(),
            'accuracy': self.acc_tracker.result()}

    @property
    def metrics(self):
        return [self.loss_tracker, self.acc_tracker]

    def get_config(self):
        """return model config"""
        config = super(AdaFaceModel, self).get_config()
        config.update({
            'n_classes': self.n_classes,
            'embedding_dim': self.fc1.units,
            'scale': self.adaface_layer.scale,
            'margin': self.adaface_layer.margin,
            'h': self.adaface_layer.h,
            'num_grad_accum': int(self.num_accum),  # convert to integer
            'dropout_rate': self.dropout.rate,
        })
        # save both config and weight
        if self.backbone is not None:
            config['backbone_config'] = self.backbone.get_config()
            # save backbone weight (serializable format)
            backbone_weights = self.backbone.get_weights()
            config['backbone_weights'] = [w.tolist() if hasattr(w, 'tolist') else w for w in backbone_weights]
        return config
    
    @classmethod
    def from_config(cls, config):
        """create model instance from config"""
        # create backbone from backbone_config
        backbone = None
        if 'backbone_config' in config:
            backbone_config = config.pop('backbone_config')
            backbone_weights = config.pop('backbone_weights', None)
            
            try:
                from keras.models import Model
                backbone = Model.from_config(backbone_config)
                
                # restore backbone weight
                if backbone_weights is not None:
                    if isinstance(backbone_weights, list):
                        # convert list to numpy array
                        weight_list = []
                        for weight in backbone_weights:
                            if isinstance(weight, list):
                                import numpy as np
                                weight_list.append(np.array(weight))
                            else:
                                weight_list.append(weight)
                        backbone.set_weights(weight_list)
                    else:
                        raise ValueError(f"backbone_weights is not a list: {type(backbone_weights)}")
                else:
                    raise ValueError("backbone_weights is not found")
                
            except Exception as e:
                raise RuntimeError(f"backbone restore failed: {e}")
        else:
            raise ValueError("backbone_config is not found")
        
        # set default values
        default_config = {
            'n_classes': 26,
            'embedding_dim': 512,
            'scale': 30.0,
            'margin': 0.4,
            'h': 0.333,
            'num_grad_accum': 1,
            'dropout_rate': 0.1,
        }
        
        # use default values for values not in config
        for key, default_value in default_config.items():
            if key not in config:
                config[key] = default_value
        
        # create model
        model = cls(backbone=backbone, **config)
        return model

    def get_inference_model(self):
        """create inference model"""
        x = self.backbone.inputs[0]
        y = self.backbone.outputs[0]
        y = self.feature_pooling(y)
        y = self.fc1(y)
        y = self.dropout(y, training=False)
        y = self.batchnorm_final(y)
        
        
        inference_model = tf.keras.Model(x, y, name='{}_embedding'.format(self.name))
        
        # model optimization setting
        inference_model.compile(optimizer='adam')  # not needed for inference, but needed for TFLite conversion
        
        return inference_model
    