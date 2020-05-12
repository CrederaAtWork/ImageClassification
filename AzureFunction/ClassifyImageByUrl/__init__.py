import logging

import azure.functions as func
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.applications.vgg19 import decode_predictions

from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient

import os
import io
import requests

connect_str = "DefaultEndpointsProtocol=https;AccountName=sigstorageaccount;AccountKey=feha7zkcJeiJJ0JONZGiYxSzUEbXp2OGec2qpt33LJpid5PyzGq2tVKG+r1wiN37gxz2RJwHaROMoiF0qPcV6w==;EndpointSuffix=core.windows.net"
local_path = "./modeldata"

def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    local_model_name = "classification_model.h5"

    # Create the BlobServiceClient object which will be used to create a container client
    blob_service_client = BlobServiceClient.from_connection_string(connect_str)
    container_name = "classification-model"
    container_client = ContainerClient.from_container_url('https://sigstorageaccount.blob.core.windows.net/classification-model')

    logging.info('Created blob service and container client')

    blobs = container_client.list_blobs()
    for blob in blobs:
        logging.info(f'Blob Name: {blob.name}')
        blob_downloader = container_client.download_blob(blob)
        blob_file = blob_downloader.readall()
        logging.info(f'Blob File as Bytes: {blob_file}')


    logging.info('Created blob client')

    # Download the blob to a local file
    # Add 'DOWNLOAD' before the .txt extension so you can see both files in the data directory
    # download_file_path = os.path.join(local_path, local_model_name)
    # logging.info("\nDownloading blob to \n\t" + download_file_path)

    # with open(download_file_path, "wb") as download_file:
    #     download_file.write(blob_client.download_blob().readall())

    # logging.info('Downloaded blob')

    # Read the image url from the request
    img_path = req.params.get('path')
    if not img_path:
        try:
            req_body = req.get_json()
        except ValueError:
            pass
        else:
            img_path = req_body.get('path')

    if img_path:

        # Constants
        img_width, img_height = 224, 224

        # Load image from url
        response = requests.get(img_path)
        raw_img = load_img(io.BytesIO(response.content), target_size=(img_width, img_height))

        # Reshape data for the model
        image = img_to_array(raw_img)
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

        return func.HttpResponse(f"PREDICTION: {label[1]} | {label[2]*100}%")
    else:
        return func.HttpResponse(
             "This HTTP triggered function executed successfully. Pass a name in the query string or in the request body for a personalized response.",
             status_code=200
        )
