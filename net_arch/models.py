import net_arch.mobilenet_v3

import tensorflow as tf


def EfficientNetB3(shape):
    model = tf.keras.applications.EfficientNetB3(
        input_shape=shape,
        classifier_activation=None,
        include_top=False,
        weights='imagenet')
    return model

def MobileNetV2(shape):
    model = tf.keras.applications.MobileNetV2(
        input_shape=shape,
        classifier_activation=None,
        include_top=False,
        weights='imagenet')
    return model

def ResNet50V2(shape):
    model = tf.keras.applications.ResNet50V2(
        input_shape=shape,
        classifier_activation=None,
        include_top=False,
        weights='imagenet')
    return model

def ResNet101V2(shape):
    model = tf.keras.applications.ResNet101V2(
        input_shape=shape,
        classifier_activation=None,
        include_top=False,
        weights='imagenet')
    return model

def ResNet152V2(shape):
    model = tf.keras.applications.ResNet152V2(
        input_shape=shape,
        classifier_activation=None,
        include_top=False,
        weights='imagenet')
    return model

def EfficientNetB7(shape):
    model = tf.keras.applications.EfficientNetB7(
        input_shape=shape,
        classifier_activation=None,
        include_top=False,
        weights='imagenet')
    return model


model_list = {
    "MobileNetV2": MobileNetV2,
    "MobileNetV3": net_arch.mobilenet_v3.MakeMobileNetV3,
    "EfficientNetB3": EfficientNetB3,
    "ResNet50": ResNet50V2,
    "ResNet101": ResNet101V2,
    "ResNet152": ResNet152V2,
    "EfficientNetB7": EfficientNetB7
}


def get_model(name, shape):
    return model_list[name](shape)
