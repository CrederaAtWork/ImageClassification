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

#import cosmos libraries
from azure.cosmos import exceptions, CosmosClient, PartitionKey

def main(myblob: func.InputStream):
    logging.info(f"Python blob trigger function processed blob \n"
                 f"Name: {myblob.name}\n"
                 f"Blob Size: {myblob.length} bytes")
    
    # Initialize the Cosmos client
    endpoint = "endpoint" #tbd
    key = 'primary_key' #tbd

    # <create_cosmos_client>
    client = CosmosClient(endpoint, key)

    # Create a database
    database_name = 'AtHomeDB'
    database = client.create_database_if_not_exists(id=database_name)

    # Create a container
    # Using a good partition key improves the performance of database operations.
    container_name = 'ImageContainer'
    container = database.create_container_if_not_exists(
        id=container_name, 
        partition_key=PartitionKey(path="/name"),
        offer_throughput=400
    )

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

    # Add items to the container
    image_items_to_create = [label]]

    # Retrieve the most likely result, e.g. highest probability
    label = label[0][0]

    logging.info(f"PREDICTION: {label[1]} | {label[2]*100}%")

    # Create Item
    for image_item in image_items_to_create:
        container.create_item(body=image_item)

    # Read items (key value lookups by partition key and id, aka point reads)
    for image in image_items_to_create:
        item_response = container.read_item(item=image['id'], partition_key=image['name'])
        request_charge = container.client_connection.last_response_headers['x-ms-request-charge']
        print('Read item with id {0}. Operation consumed {1} request units'.format(item_response['id'], (request_charge)))

    # Don't think we need this for our purposes, but included jic
    # Query these items using the SQL query syntax. 
    # Specifying the partition key value in the query allows Cosmos DB to retrieve data only from the relevant partitions, which improves performance

    # query = "SELECT * FROM c WHERE c.name IN ('sofa', 'chair')"
    
    # items = list(container.query_items(
    #     query=query,
    #     enable_cross_partition_query=True
    # ))

    # request_charge = container.client_connection.last_response_headers['x-ms-request-charge']

    # print('Query returned {0} items. Operation consumed {1} request units'.format(len(items), request_charge))