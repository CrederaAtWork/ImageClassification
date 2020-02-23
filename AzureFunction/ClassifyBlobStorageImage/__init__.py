import logging

import azure.functions as func

from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.applications.vgg19 import decode_predictions

import base64
import io

import numpy as np
from PIL import Image


def main(myblob: func.InputStream):
    logging.info(f"Python blob trigger function processed blob \n"
                 f"Name: {myblob.name}\n"
                 f"Blob Size: {myblob.length} bytes")
    
    # Constants
    img_width, img_height = 224, 224

    # Load image from blob
    bytes_image = myblob.read() 
    raw_image = load_img(io.BytesIO(bytes_image), target_size=(img_width, img_height))

    # Reshape data for the model
    image = img_to_array(raw_image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    
    # Prepare the image for the VGG model
    image = preprocess_input(image)

    # Load base model
    if K.image_data_format() == 'channels_first':
        input_shape = (3, img_width, img_height)
    else:
        input_shape = (img_width, img_height, 3)
    
    test_model = VGG19(input_shape = input_shape, weights='imagenet')

    # Perform image classification
    prediction = test_model.predict(image)

    # Convert the probabilities to class labels
    label = decode_predictions(prediction)
    # Retrieve the most likely result, e.g. highest probability
    label = label[0][0]

    logging.info(f"PREDICTION: {label[1]} | {label[2]*100}%")
