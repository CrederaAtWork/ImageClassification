import logging

import azure.functions as func

from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient

import tensorflow as tf

import os
import io
import h5py

connect_str = os.environ["CONTAINER_CONNECTION_STRING"]
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

        with h5py.File(blob.content) as h5_file:
            model = tf.keras.models.load_model(h5_file)
            logging.info(f'MMMModel: {model}')
        # blob_downloader = container_client.download_blob(blob)
        # blob_file = blob_downloader.readall()
        # logging.info(f'Blob File as Bytes: {blob_file}')

    
    logging.info('Created blob client')

    path = req.params.get('path')
    if not path:
        try:
            req_body = req.get_json()
        except ValueError:
            pass
        else:
            path = req_body.get('path')

    if path:
        return func.HttpResponse(f"Hello, {path}. This HTTP triggered function executed successfully.")
    else:
        return func.HttpResponse(
             "This HTTP triggered function executed successfully. Pass a name in the query string or in the request body for a personalized response.",
             status_code=200
        )
