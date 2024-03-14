import os
import json

from PIL import Image
import numpy as np
import requests

from .params import VGG_CLASS_LABEL_LINK

def load_image(path:str, size:int=(224, 224), do_norm:bool=True):
    mean = [103.939, 116.779, 123.68]
    image = Image.open(path).resize(size).convert("RGB")
    image = np.array(image).astype(np.float32)
    if do_norm:
        image = image[..., ::-1]
        image[..., 0] -= mean[0]
        image[..., 1] -= mean[1]
        image[..., 2] -= mean[2]
    return np.expand_dims(image, axis=0)

def decode_probabilities(prob, top:int=5):
    if os.path.exists('resources/imagenet_class_index.json'):
        with open('resources/imagenet_class_index.json') as f:
            class_index = json.load(f)
    else:
        os.makedirs('resources', exist_ok=True)
        class_index = requests.get(VGG_CLASS_LABEL_LINK).json()
        with open('resources/imagenet_class_index.json', 'w') as f:
            json.dump(class_index, f)
    
    class_index = {int(k): v[1] for k, v in class_index.items()}
    top_indices = np.argsort(prob)[::-1][:top]
    ret = [(class_index[i], prob[i]) for i in top_indices]
    return ret
