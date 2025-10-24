#!/usr/bin/env python3
"""
keras model to TFLite conversion script
"""
# python convert_tflite/main.py --input_path checkpoints/ResNet50_adaface_251001_2/best_inference.keras --output checkpoints/ResNet50_adaface_251001_2/best_inference_56.tflite

import os
import sys
import argparse
import tensorflow as tf
import numpy as np

from train.config import config
from train.custom_models.cosface_model import CosFaceModel
from train.custom_models.adaface_model import AdaFaceModel
from train.custom_models.angular_margin_model import AngularMarginModel

def load_keras_model(model_path):
    """Load Keras model"""
    try:
        # define custom objects
        custom_objects = {
            'CosFaceModel': CosFaceModel,
            'AdaFaceModel': AdaFaceModel,
            'AngularMarginModel': AngularMarginModel,
        }
        
        # load model (without compilation)
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
        print(f"✅ Keras model load success: {model_path}")
        
        # compile model after weights load
        model.compile()
        
        return model
    except Exception as e:
        print(f"❌ Model load failed: {e}")
        print("🔄 Try to reconstruct model with weights only...")
        
        try:
            # create inference model only
            inference_model = create_inference_model_from_weights(model_path)
            if inference_model:
                print(f"✅ Inference model creation success")
                return inference_model
        except Exception as e2:
            print(f"❌ Inference model creation failed: {e2}")
        
        return None

def create_inference_model_from_weights(model_path):
    """Create inference model from weights"""
    try:
        # extract model information from config
        loss_type = config['loss']
        n_classes = config['num_identity']
        embedding_dim = config['embedding_dim']
        
        # create backbone model
        from train.main import build_backbone_model
        backbone, _ = build_backbone_model(config)
        
        # create model based on loss type
        if loss_type == 'CosFace':
            model = CosFaceModel(
                backbone=backbone,
                n_classes=n_classes,
                embedding_dim=embedding_dim,
                scale=config['loss_param']['CosFace']['scale'],
                margin=config['loss_param']['CosFace']['margin'],
            )
        elif loss_type == 'AdaFace':
            model = AdaFaceModel(
                backbone=backbone,
                n_classes=n_classes,
                embedding_dim=embedding_dim,
                scale=config['loss_param']['AdaFace']['scale'],
                margin=config['loss_param']['AdaFace']['margin'],
                h=config['loss_param']['AdaFace']['h'],
                dropout_rate=0.1
            )
        elif loss_type == 'AngularMargin':
            model = AngularMarginModel(
                backbone=backbone,
                n_classes=n_classes,
                embedding_dim=embedding_dim,
                scale=config['loss_param']['AngularMargin']['scale'],
                margin=config['loss_param']['AngularMargin']['margin']
            )
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")
        
        # load weights based on file extension
        if model_path.endswith('.keras'):
            # .keras file, skip by_name
            model.load_weights(model_path, skip_mismatch=True)
            print("✅ Weight load success from .keras file")
        else:
            # .h5 file, use by_name
            model.load_weights(model_path, by_name=True, skip_mismatch=True)
            print("✅ Weight load success from .h5 file (using by_name)")
        
        # compile model after weights load
        model.compile()
        
        # extract inference model
        inference_model = model.get_inference_model()
        return inference_model
        
    except Exception as e:
        print(f"❌ Inference model creation failed: {e}")
        return None

def get_inference_model(keras_model):
    """Extract inference model"""
    try:
        if hasattr(keras_model, 'get_inference_model'):
            # Custom model, extract inference model
            inference_model = keras_model.get_inference_model()
            print("✅ Inference model extraction completed")
        else:
            # regular Keras model, use as is
            inference_model = keras_model
            print("✅ Keras model used directly")
        
        return inference_model
    except Exception as e:
        print(f"❌ Inference model extraction failed: {e}")
        return None

def convert_to_tflite(model, output_path):
    """TFLite conversion - create fully compatible model without TF Select"""
    try:
        # create TFLite converter
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        
        # apply default optimizations
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # don't use TF Select - create fully compatible model
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            # tf.lite.OpsSet.TFLITE_BUILTINS_INT8
        ]
        
        # force float32 type
        converter.target_spec.supported_types = [tf.float32]
        
        # run conversion
        tflite_model = converter.convert()
        
        # save file
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        
        print(f"✅ TFLite conversion success: {output_path}")
        
        # print model size
        model_size = os.path.getsize(output_path) / (1024 * 1024)
        print(f"📊 Model size: {model_size:.2f} MB")
        
        return True
        
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        return False

def test_tflite_model(tflite_path):
    """Test TFLite model"""
    try:
        # load TFLite interpreter
        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()
        
        # input/output information
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print(f"\n📋 TFLite model information:")
        print(f"  Input shape: {input_details[0]['shape']}")
        print(f"  Input type: {input_details[0]['dtype']}")
        print(f"  Output shape: {output_details[0]['shape']}")
        print(f"  Output type: {output_details[0]['dtype']}")
        
        # test inference
        test_input = np.random.random((1, 112, 112, 3)).astype(np.float32)
        interpreter.set_tensor(input_details[0]['index'], test_input)
        interpreter.invoke()
        
        output = interpreter.get_tensor(output_details[0]['index'])
        print(f"  Test inference success, output shape: {output.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ TFLite test failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Convert Best.keras to TFLite')
    parser.add_argument('--input_path', type=str, required=True,
                       help='Path to best.keras file')
    parser.add_argument('--output_path', type=str, 
                       help='Output TFLite file path (auto-generated if not specified)')
    parser.add_argument('--test_model', action='store_true',
                       help='Test the converted TFLite model')
    
    args = parser.parse_args()
    
    # auto-generate output path
    if not args.output_path:
        base_name = os.path.splitext(os.path.basename(args.input_path))[0]
        args.output_path = f"{base_name}.tflite"
    
    print(f"🚀 {args.input_path} to TFLite conversion...")
    print(f"📁 Output: {args.output_path}")
    
    # 1. Load Keras model
    keras_model = load_keras_model(args.input_path)
    if keras_model is None:
        return
    
    # 2. Extract inference model
    inference_model = get_inference_model(keras_model)
    if inference_model is None:
        return
    
    # 3. Print model information
    print(f"\n📋 Model information:")
    print(f"  Input shape: {inference_model.input_shape}")
    print(f"  Output shape: {inference_model.output_shape}")
    print(f"  Parameters: {inference_model.count_params():,}")
    
    # 4. TFLite conversion
    success = convert_to_tflite(inference_model, args.output_path)
    
    # 5. Test (optional)
    if success and args.test_model:
        test_tflite_model(args.output_path)
    
    if success:
        print(f"\n✅ Conversion completed: {args.output_path}")
    else:
        print(f"\n❌ Conversion failed!")

if __name__ == '__main__':
    main()
