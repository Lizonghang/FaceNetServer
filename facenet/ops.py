import numpy as np
import cv2
import StringIO
from PIL import Image


def read_image(requests):
    """load image only"""
    im = requests.FILES.get('image')
    im = StringIO.StringIO(im.read())
    im = Image.open(im)
    return np.array(im)


def read_and_resize_image(requests):
    """load image and resize"""
    im = requests.FILES.get('image').read()
    im = StringIO.StringIO(im)
    im = Image.open(im)
    im = im.resize((160, 160))
    return np.array(im)


def read_bytes(requests, size, reverse_rgb=False):
    """convert bytes to opencv image"""
    im = requests.FILES.get('image_bytes').read()
    im = Image.frombytes('RGB', size, im)
    im = np.array(im)
    if reverse_rgb:
        im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
    return im


def normalize_image(im):
    im = np.array(im)
    im = (im - np.mean(im)) / np.maximum(np.std(im), 1.0 / np.sqrt(im.size))
    im = im.reshape((-1, 160, 160, 3))
    return im
