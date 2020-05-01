import logging
import os

import azure.functions as func
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
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


def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')
    img_width, img_height = 224, 224

    conn_str = os.environ['AZURE_STORAGE_CONNECTION_STRING']
    #account_key = os.environ['Storage_Account_Key']
    container_name = 'furnitureimages'

    blob_service_client = BlobServiceClient.from_connection_string(conn_str)
    container_client = blob_service_client.get_container_client(container_name)

    blobs = container_client.list_blobs()
    for blob in blobs:
        logging.info(f'blob.name: {blob.name}')
        # Load image from blob
        blob_stream = container_client.download_blob(blob)
        bytes_image = blob_stream.content_as_bytes()
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






    name = req.params.get('name')
    if not name:
        try:
            req_body = req.get_json()
        except ValueError:
            pass
        else:
            name = req_body.get('name')

    if name:
        return func.HttpResponse(f"Hello, {name}. This HTTP triggered function executed successfully.")
    else:
        return func.HttpResponse(
             "This HTTP triggered function executed successfully. Pass a name in the query string or in the request body for a personalized response.",
             status_code=200
        )
