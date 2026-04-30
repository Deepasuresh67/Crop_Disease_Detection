from django.shortcuts import render
from .models import UploadedImage

import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import requests
from io import BytesIO

from .forms import ImageUploadForm, ImageURLForm
from azure.storage.blob import BlobServiceClient

import os
import uuid

# 🔥 IMPORTANT: add your connection string here for local testing
os.environ["AZURE_STORAGE_CONNECTION_STRING"] = "DefaultEndpointsProtocol=https;AccountName=cropdisease1;AccountKey=XZby8VonbBM5Uhtp8xNSUck9Tw5tACj9+mqioDwvAP3tJvTkciUQgM+ZcCHi+s1swdm1QRowwVZX+ASt3ZBPqQ==;EndpointSuffix=core.windows.net"

# Base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load model
model_path = os.path.join(BASE_DIR, "detection", "model.tflite")
labels_path = os.path.join(BASE_DIR, "detection", "labels.txt")

interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Load class names
with open(labels_path, "r") as file:
    class_names = [line.strip() for line in file.readlines()]


# ✅ Upload to Azure Blob Storage
def upload_to_blob(file):
    connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    container_name = "leaf-images"

    blob_service_client = BlobServiceClient.from_connection_string(connection_string)

    # 🔥 unique filename (prevents overwrite)
    blob_name = str(uuid.uuid4()) + "_" + file.name

    blob_client = blob_service_client.get_blob_client(
        container=container_name,
        blob=blob_name
    )

    file.seek(0)  # 🔥 VERY IMPORTANT

    blob_client.upload_blob(file, overwrite=True)

    image_url = f"https://{blob_service_client.account_name}.blob.core.windows.net/{container_name}/{blob_name}"

    return image_url


# Views
def home(request):
    return render(request, 'detection/index.html')


def upload(request):
    return render(request, 'detection/upload.html')


def weather_view(request):
    return render(request, 'detection/weather.html')


# ✅ Upload + Predict
def upload_image(request):

    if request.method == 'POST':
        upload_form = ImageUploadForm(request.POST, request.FILES)

        if upload_form.is_valid():

            image_file = request.FILES['image']

            # Upload to Azure
            image_url = upload_to_blob(image_file)

            # 🔥 reset pointer before prediction
            image_file.seek(0)

            result = predict_image(image_file)

            if 'error' in result:
                return render(request, 'detection/upload_image.html', {
                    'upload_form': upload_form,
                    'error': result['error']
                })

            return render(request, 'detection/result.html', {
                'class_name': result['class_name'],
                'confidence': result['confidence'],
                'image_url': image_url
            })

        return render(request, 'detection/upload_image.html', {
            'upload_form': upload_form,
            'error': 'Invalid form submission'
        })

    upload_form = ImageUploadForm()

    return render(request, 'detection/upload_image.html', {
        'upload_form': upload_form
    })


# URL Prediction
def enter_url(request):
    if request.method == 'POST':
        url_form = ImageURLForm(request.POST)

        if url_form.is_valid():
            image_url = url_form.cleaned_data['image_url']

            result = predict_image(image_url)

            if 'error' in result:
                return render(request, 'detection/enter_url.html', {
                    'url_form': url_form,
                    'error': result['error']
                })

            return render(request, 'detection/result.html', {
                'class_name': result['class_name'],
                'confidence': result['confidence'],
                'image_url': image_url
            })

    else:
        url_form = ImageURLForm()

    return render(request, 'detection/enter_url.html', {
        'url_form': url_form,
    })


# ✅ Prediction Function
def predict_image(image_path_or_url):
    try:

        if isinstance(image_path_or_url, str) and image_path_or_url.startswith('http'):
            response = requests.get(image_path_or_url)
            image = Image.open(BytesIO(response.content)).convert("RGB")
        else:
            image = Image.open(image_path_or_url).convert("RGB")

        size = (224, 224)
        image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

        image_array = np.asarray(image)
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        interpreter.set_tensor(input_details[0]['index'], [normalized_image_array])
        interpreter.invoke()

        output_data = interpreter.get_tensor(output_details[0]['index'])

        index = np.argmax(output_data)
        class_name = class_names[index]
        confidence_score = output_data[0][index] * 100

        return {
            'class_name': class_name,
            'confidence': confidence_score
        }

    except Exception as e:
        return {'error': str(e)}