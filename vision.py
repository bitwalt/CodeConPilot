"""Open and upscale an image using PIL."""

import cv2
import numpy as np
import PIL.Image
import PIL.ImageOps
import PIL.ImageDraw
import PIL.ImageFont
import PIL.ImageFilter

def load_image(filename):
    """Load an image from disk."""
    img = cv2.imread(filename)
    if img is None:
        raise IOError("Could not read image file {}".format(filename))
    return img  

def upscale_image(img):
    """Upscale an image to a higher resolution."""
    return cv2.pyrUp(img)
