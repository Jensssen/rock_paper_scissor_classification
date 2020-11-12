import boto3
import numpy as np
from PIL import Image
from io import BytesIO
import os 
import json

import sagemaker
from sagemaker.tensorflow.model import TensorFlowModel

s3_client = boto3.client('s3')

def from_s3(bucket, key):
    file_byte_string = s3_client.get_object(Bucket=bucket, Key=key)['Body'].read()
    return Image.open(BytesIO(file_byte_string))


def inference_trigger(event, context):
    bucket = "rockpaperscissor903684e44d4d41a89bf150937ea98fd143614-dev"
    key = "public/inference/inference.png"
    ENDPOINT_NAME = os.environ['ENDPOINT_NAME']
    sagemaker_session = sagemaker.Session(default_bucket=bucket)

    # Download and prepare input image
    image = from_s3(bucket, key)

    image = image.resize((224,224))
    image = np.array(image)/255
    image = np.expand_dims(image, axis=0)

    # Call Sagemaker Endpoint on image
    predictor=sagemaker.tensorflow.model.TensorFlowPredictor(ENDPOINT_NAME, sagemaker_session)

    result = predictor.predict(image)

    result["argmax"] = str(np.argmax(result["predictions"]))

    result_string = str(result["predictions"][0][0]) + "," + str(result["predictions"][0][1]) + "," + str(result["predictions"][0][2]) + "," + str(result["argmax"])

    s3_client.put_object(Bucket=bucket, Key="public/inference_result/inference_result.json", Body = json.dumps(result))
    s3_client.put_object(Bucket=bucket, Key="public/inference_result/inference_result.txt", Body = json.dumps(result_string))