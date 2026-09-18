"""Paths and model loading shared by the command line and UI."""
from pathlib import Path
import os
import json

ROOT = Path(__file__).resolve().parent

def local_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path

def load_settings():
    from dotenv import load_dotenv
    load_dotenv(ROOT / '.env')

def load_classifier(model_path=None, labels_path=None):
    load_settings()
    mp = local_path(model_path or os.getenv('MODEL_PATH', 'models/fruit_classifier_model.keras'))
    lp = local_path(labels_path or os.getenv('LABELS_PATH', 'models/fruit_class_names.json'))
    if not mp.is_file() or not lp.is_file():
        raise FileNotFoundError(
            f'Chưa có model hoặc nhãn: {mp} ; {lp}. '
            'Chạy python train.py hoặc chép model + JSON nhãn và sửa .env.')
    labels = json.loads(lp.read_text(encoding='utf-8'))
    if (not isinstance(labels, list) or not labels or
            any(not isinstance(x, str) or not x for x in labels) or
            len(set(labels)) != len(labels)):
        raise ValueError('JSON nhãn phải là danh sách tên lớp duy nhất theo thứ tự huấn luyện.')
    import tensorflow as tf
    from tensorflow.keras.applications.efficientnet import preprocess_input
    model = tf.keras.models.load_model(
        mp, compile=False, custom_objects={'preprocess_input': preprocess_input})
    if model.output_shape[-1] != len(labels):
        raise ValueError('Số lớp của model không khớp JSON nhãn.')
    if tuple(model.input_shape[1:]) != (224, 224, 3):
        raise ValueError('Ứng dụng này yêu cầu model có input (224, 224, 3).')
    return model, labels

def image_batch(img):
    import numpy as np
    from PIL import ImageOps, Image
    img = ImageOps.exif_transpose(img).convert('RGB')
    # Giữ pixel 0..255; EfficientNet có Rescaling bên trong.
    return np.asarray(img.resize((224, 224), Image.Resampling.BILINEAR), dtype=np.float32)[None, ...]
