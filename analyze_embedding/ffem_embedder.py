import numpy as np
import cv2
import logging
import tensorflow as tf
import math
from typing import Dict, List, Tuple
import time
import os
import sys

# logging settings
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FFEMTfliteEmbedder:
    def __init__(self, model_path, device="cpu", input_shape=(112, 112), use_normalization=True):
        """
        Class to extract face embedding using FFEM TFLite model.
        
        Args:
            model_path (str): TFLite model file path
            device (str): execution device ("cpu" or "gpu")
            input_shape (tuple): model input size (H, W), default (112, 112)
            use_normalization (bool): input image normalization (0~1)
        """
        self.model_path = model_path
        self.device = device.lower()
        self.input_shape = input_shape
        self.use_normalization = use_normalization
        self.interpreter = self._setup_interpreter()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        logger.info(f"Loaded FFEM model {self.model_path} on {self.device}")
        logger.info(f"Input shape: {self.input_details[0]['shape']}, Output shape: {self.output_details[0]['shape']}")

    def _setup_interpreter(self):
        """Setup TFLite interpreter"""
        try:
            if self.device == "cpu":
                interpreter = tf.lite.Interpreter(model_path=self.model_path)
            elif self.device == "gpu":
                interpreter = tf.lite.Interpreter(
                    model_path=self.model_path,
                    experimental_delegates=[tf.lite.load_delegate('./libtensorflowlite_gpu_delegate.so')]
                )
            else:
                raise ValueError(f"Unsupported device: {self.device}")
            interpreter.allocate_tensors()
            return interpreter
        except Exception as e:
            logger.error(f"Failed to setup interpreter: {e}")
            raise

    def preprocess_image(self, image):
        """
        Preprocess input image to model input size (1, 112, 112, 3).
        
        Args:
            image (np.ndarray): input image, (H, W, 3), BGR format
            
        Returns:
            np.ndarray: preprocessed image, (1, 112, 112, 3)
            float: preprocessing time (seconds)
        """
        start_time = time.time()
        
        if not isinstance(image, np.ndarray) or len(image.shape) != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected image shape (H, W, 3), got {image.shape if isinstance(image, np.ndarray) else type(image)}")

        target_h, target_w = self.input_shape

        # resize
        resized_image = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

        # BGR -> RGB conversion
        img_array = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)

        # normalization (0~1)
        if self.use_normalization:
            img_array = img_array.astype(np.float32) / 255.0
        else:
            img_array = img_array.astype(np.float32)

        # add batch dimension (1, H, W, C)
        input_data = np.expand_dims(img_array, axis=0)

        return input_data, time.time() - start_time

    def run_inference(self, input_data):
        """
        Run model inference.
        
        Args:
            input_data (np.ndarray): preprocessed input data, (1, 112, 112, 3)
            
        Returns:
            np.ndarray: embedding output, (1, 512)
            float: inference time (seconds)
        """
        if input_data.shape != (1, *self.input_shape, 3):
            raise ValueError(f"Expected input shape (1, {self.input_shape[0]}, {self.input_shape[1]}, 3), got {input_data.shape}")

        start_time = time.time()
        self.interpreter.set_tensor(self.input_details[0]["index"], input_data)
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(self.output_details[0]["index"])
        inference_time = time.time() - start_time

        return output_data, inference_time

    def get_embedding(self, image):
        """
        Extract face embedding from input image (full process).
        
        Args:
            image (np.ndarray): input image, (H, W, 3), BGR format
            
        Returns:
            np.ndarray: face embedding, (1, 512)
            dict: preprocessing and inference time information
        """
        try:
            # preprocessing
            input_data, preprocess_time = self.preprocess_image(image)
            
            # inference
            embedding, inference_time = self.run_inference(input_data)
            
            # check output size
            if embedding.shape != (1, 512):
                logger.warning(f"Unexpected output shape: {embedding.shape}, expected (1, 512)")
            
            return embedding, {
                "preprocess_time": preprocess_time,
                "inference_time": inference_time,
                "total_time": preprocess_time + inference_time
            }
        except Exception as e:
            logger.error(f"Error in get_embedding: {e}")
            raise

class FaceSearchModule:
    """Python ported FaceSearchModule (based on Java original, supports multiple reference images)"""
    def __init__(self, face_list: Dict[str, np.ndarray] = None):
        self.registered_faces = {}
        if face_list:
            self.init_face_list(face_list)

    def init_face_list(self, face_list: Dict[str, List[np.ndarray]]):
        """Initialize registered face list and L2 normalization (multiple embedding averaging)"""
        self.registered_faces = {}
        for key, embeddings in face_list.items():
            if not embeddings:
                logger.warning(f"No embeddings provided for {key}")
                continue
            # multiple embedding averaging
            avg_embedding = np.mean(embeddings, axis=0)
            self.registered_faces[key] = self._l2_norm(avg_embedding)
            logger.info(f"Averaged {len(embeddings)} embeddings for {key}, shape: {avg_embedding.shape}")

    def remove(self, key: str):
        """Remove registered face"""
        self.registered_faces.pop(key, None)

    def add(self, key: str, embedding: np.ndarray):
        """Add new face (L2 normalization after)"""
        self.registered_faces[key] = self._l2_norm(embedding)

    def search(self, embedding: np.ndarray, threshold: float) -> Tuple[str, float]:
        """Search face by embedding, return matching ID and angle"""
        best_sim = 1e3
        best_id = ""
        embedding = self._l2_norm(embedding)
        
        for key, reg_embedding in self.registered_faces.items():
            dot = np.dot(reg_embedding, embedding)
            angle = math.acos(np.clip(dot, -1.0, 1.0)) * 180.0 / math.pi
            if angle < best_sim:
                best_sim = angle
                best_id = key
        
        logger.info(f"Search result - Name: {best_id}, Angle: {best_sim:.2f}")
        if best_sim <= threshold:
            return best_id, best_sim
        return "", best_sim

    def get_face_list(self) -> List[str]:
        """Return registered face key list"""
        return list(self.registered_faces.keys())

    def _l2_norm(self, embedding: np.ndarray) -> np.ndarray:
        """L2 normalization"""
        norm = np.sqrt(np.sum(embedding ** 2))
        if norm == 0:
            logger.warning("Zero norm detected in L2 normalization")
            return embedding
        return embedding / norm


class FFEMKerasEmbedder:
    def __init__(self, model_path, device="cpu", input_shape=(112, 112), use_normalization=True):
        """
        Class to extract face embedding using FFEM Keras(.keras/.h5) model.
        """
        self.model_path = model_path
        self.device = device.lower()
        self.input_shape = input_shape
        self.use_normalization = use_normalization
        # ensure project root on sys.path to import custom layers/models
        try:
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
            if project_root not in sys.path:
                sys.path.insert(0, project_root)
        except Exception as e:
            logger.warning(f"sys.path setup failed: {e}")

        # build custom_objects mapping for deserialization
        custom_objects = {}
        try:
            from train.layers.norm_aware_pooling_layer import NormAwarePoolingLayer
            custom_objects['NormAwarePoolingLayer'] = NormAwarePoolingLayer
            custom_objects['Custom>NormAwarePoolingLayer'] = NormAwarePoolingLayer
        except Exception as e:
            logger.warning(f"Could not import NormAwarePoolingLayer: {e}")
        try:
            from train.custom_models.angular_margin_model import AngularMarginModel
            custom_objects['AngularMarginModel'] = AngularMarginModel
        except Exception as e:
            logger.warning(f"Could not import AngularMarginModel: {e}")
        try:
            from train.custom_models.cosface_model import CosFaceModel
            custom_objects['CosFaceModel'] = CosFaceModel
        except Exception as e:
            logger.warning(f"Could not import CosFaceModel: {e}")
        try:
            from train.layers.angular_margin_layer import AngularMarginLayer
            custom_objects['AngularMarginLayer'] = AngularMarginLayer
        except Exception as e:
            logger.warning(f"Could not import AngularMarginLayer: {e}")
        try:
            from train.layers.cosface_layer import CosFaceLayer
            custom_objects['CosFaceLayer'] = CosFaceLayer
        except Exception as e:
            logger.warning(f"Could not import CosFaceLayer: {e}")

        # attempt to load model with custom_objects
        last_err = None
        for safe_mode in (True, False):
            try:
                self.model = tf.keras.models.load_model(
                    self.model_path,
                    custom_objects=custom_objects if custom_objects else None,
                    compile=False,
                    safe_mode=safe_mode
                )
                break
            except Exception as e:
                last_err = e
                logger.warning(f"Keras load_model failed (safe_mode={safe_mode}): {e}")
                self.model = None
        if self.model is None:
            logger.error(f"Failed to load Keras model with custom objects: {last_err}")
            raise last_err
        logger.info(f"Loaded Keras FFEM model {self.model_path}")

    def preprocess_image(self, image):
        start_time = time.time()
        if not isinstance(image, np.ndarray) or len(image.shape) != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected image shape (H, W, 3), got {image.shape if isinstance(image, np.ndarray) else type(image)}")
        target_h, target_w = self.input_shape
        resized_image = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        img_array = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
        if self.use_normalization:
            img_array = img_array.astype(np.float32) / 255.0
        else:
            img_array = img_array.astype(np.float32)
        input_data = np.expand_dims(img_array, axis=0)
        return input_data, time.time() - start_time

    def run_inference(self, input_data):
        start_time = time.time()
        # keras model assumes NHWC channel order
        outputs = self.model(input_data, training=False)
        if isinstance(outputs, (list, tuple)):
            outputs = outputs[0]
        output_data = outputs.numpy() if hasattr(outputs, 'numpy') else np.array(outputs)
        inference_time = time.time() - start_time
        return output_data, inference_time

    def get_embedding(self, image):
        try:
            input_data, preprocess_time = self.preprocess_image(image)
            embedding, inference_time = self.run_inference(input_data)
            # try to ensure (1, 512) shape
            if embedding.ndim == 1:
                embedding = embedding[None, :]
            if embedding.shape[0] != 1:
                embedding = embedding[:1]
            return embedding, {
                "preprocess_time": preprocess_time,
                "inference_time": inference_time,
                "total_time": preprocess_time + inference_time
            }
        except Exception as e:
            logger.error(f"Error in get_embedding (Keras): {e}")
            raise

    def preprocess_images(self, images):
        """
        Preprocess multiple images at once and return batch input.
        images: List[np.ndarray] or np.ndarray of shape (N, H, W, 3)
        """
        start_time = time.time()
        batch = []
        for img in images:
            if not isinstance(img, np.ndarray) or len(img.shape) != 3 or img.shape[2] != 3:
                raise ValueError(f"Expected image shape (H, W, 3), got {img.shape if isinstance(img, np.ndarray) else type(img)}")
            target_h, target_w = self.input_shape
            resized_image = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            img_array = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
            if self.use_normalization:
                img_array = img_array.astype(np.float32) / 255.0
            else:
                img_array = img_array.astype(np.float32)
            batch.append(img_array)
        input_data = np.stack(batch, axis=0)
        return input_data, time.time() - start_time

    def get_embeddings_batch(self, images):
        """
        Convert batch images to embeddings at once.
        Returns: (N, D) embedding and timing dict
        """
        try:
            input_batch, preprocess_time = self.preprocess_images(images)
            outputs, inference_time = self.run_inference(input_batch)
            if outputs.ndim == 1:
                outputs = outputs[None, :]
            return outputs, {
                "preprocess_time": preprocess_time,
                "inference_time": inference_time,
                "total_time": preprocess_time + inference_time
            }
        except Exception as e:
            logger.error(f"Error in get_embeddings_batch (Keras): {e}")
            raise