import argparse
import os

from train.layers.group_aware_layer import GroupAwareLayer
from train.layers.norm_aware_pooling_layer import NormAwarePoolingLayer
from train.layers.angular_margin_layer import AngularMarginLayer
from train.layers.center_margin_layer import CenterMarginLayer

import tensorflow as tf

keras_custom_objects = {
    'GroupAwareLayer': GroupAwareLayer,
    'NormAwarePoolingLayer': NormAwarePoolingLayer,
    'AngularMarginLayer': AngularMarginLayer,
    'CenterMarginLayer': CenterMarginLayer
}

def convert_tflite_int8(model, calb_data, output_name, quant_level=0):
    """
    quant_level == 0:
        weights only quantzation, no requires calibration data.
    quant_level == 1:
        Full quantization for supported operators.
        It remains float for not supported operators.
    quant_level == 2:
        Full quantization for all operators.
        It can not be converted if the model contains not supported operators.
    """
    if hasattr(model, 'layers'):
        for layer in model.layers:
            if hasattr(layer, 'dtype_policy'):
                layer.dtype_policy = tf.keras.mixed_precision.Policy('float32')
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float32]
    def representative_dataset_gen():
        for n, (x, _ )in enumerate(calb_data.take(5000)):
            if n % 100 == 0:
                print(f"Processing calibration data: {n}")
            if x.shape[0] > 1:
                x = x[:1]
            x = tf.cast(x, tf.float32)
            yield [x]
    if quant_level == 1:
        converter.representative_dataset = representative_dataset_gen
        converter.inference_input_type = tf.float32
        converter.inference_output_type = tf.float32
    elif quant_level == 2:
        converter.representative_dataset = representative_dataset_gen
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8
    
    try:
        tflite_quant_model = converter.convert()
        with open(output_name, 'wb') as f:
            f.write(tflite_quant_model)
        print(f"TFLite model successfully converted and saved to: {output_name}")
    except Exception as e:
        print(f"TFLite conversion failed: {e}")
        print("Trying fallback conversion without quantization...")
        
        # Fallback
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.target_spec.supported_types = [tf.float32]
        
        try:
            tflite_model = converter.convert()
            fallback_name = output_name.replace('.tflite', '_fallback.tflite')
            with open(fallback_name, 'wb') as f:
                f.write(tflite_model)
            print(f"Fallback TFLite model saved to: {fallback_name}")
        except Exception as e2:
            print(f"Fallback conversion also failed: {e2}")
            raise e2


def input_pipeline(dataset_file, input_shape):
    def _read_tfrecord(serialized):
        description = {
            'jpeg': tf.io.FixedLenFeature((), tf.string),
            'label': tf.io.FixedLenFeature((), tf.int64),
            'x1': tf.io.FixedLenFeature((), tf.int64),
            'y1': tf.io.FixedLenFeature((), tf.int64),
            'x2': tf.io.FixedLenFeature((), tf.int64),
            'y2': tf.io.FixedLenFeature((), tf.int64)
        }
        example = tf.io.parse_single_example(serialized, description)
        image = tf.io.decode_jpeg(example['jpeg'], channels=3)
        label = example['label']
        box = [
            tf.cast(example['y1'], tf.float32),
            tf.cast(example['x1'], tf.float32),
            tf.cast(example['y2'], tf.float32),
            tf.cast(example['x2'], tf.float32)]
        return image, label, box

    def _load_and_preprocess_image(image, label, box):
        # shape = [Height, Width, Channel]
        shape = tf.shape(image)
        # shape = [Height, Height, Width, Width]
        shape = tf.repeat(shape, [2, 2, 0])
        # shape = [Height, Width, Height, Width]
        shape = tf.scatter_nd([[0], [2], [1], [3]], shape, tf.constant([4]))
        # Normalize [y1, x1, y2, x2] box by width and height.
        box /= tf.cast(shape, tf.float32)
        image = tf.cast(image, tf.float32)
        return image, label, box


    def _normalize(x: tf.Tensor):
        # Normalize images to the range [0, 1].
        return x / 255.

    test_ds = tf.data.TFRecordDataset(dataset_file)
    test_ds = test_ds.map(_read_tfrecord)
    test_ds = test_ds.map(_load_and_preprocess_image)
    test_ds = test_ds.map(
        lambda x, label, box: (tf.image.crop_and_resize([x], [box], [0], input_shape)[0], label))
    test_ds = test_ds.batch(1)
    test_ds = test_ds.map(lambda img, label : (_normalize(img), label))

    return test_ds


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--keras_model', type=str, required=True,
        help='trained keras model path')
    parser.add_argument('--dataset', type=str, required=True,
        help='calibration dataset (tfrecord)')
    parser.add_argument('--image_size', type=str, required=True,
        help='image width and height. ex) 112,112')
    parser.add_argument('--quant_level', type=int, required=False,
        default=0, help='quantization level 0 ~ 2')
    args = parser.parse_args()
    img_size = args.image_size.split(',')
    width = int(img_size[0])
    height = int(img_size[1])
    output_name = os.path.splitext(args.keras_model)[0] + '.tflite'
    quant_level = args.quant_level
    dataset = input_pipeline(args.dataset, (width, height))
    net = tf.keras.models.load_model(args.keras_model, custom_objects=keras_custom_objects)
    convert_tflite_int8(net, dataset, output_name, quant_level)
