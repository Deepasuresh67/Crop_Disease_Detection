from django.shortcuts import render, redirect
from django.core.files.storage import FileSystemStorage
from .models import UploadedImage
import tensorflow as tf
from PIL import Image, ImageOps, UnidentifiedImageError
import numpy as np
import requests
from io import BytesIO
from .forms import ImageUploadForm, ImageURLForm

import os


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model_path = os.path.join(BASE_DIR, "detection", "model.tflite")
labels_path = os.path.join(BASE_DIR, "detection", "labels.txt")

interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

def upload_to_blob(file):
    connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    container_name = "leaf-images"

    blob_service_client = BlobServiceClient.from_connection_string(connection_string)

    blob_client = blob_service_client.get_blob_client(
        container=container_name,
        blob=file.name
    )

    blob_client.upload_blob(file, overwrite=True)

    # generate image URL
    image_url = f"https://{blob_service_client.account_name}.blob.core.windows.net/{container_name}/{file.name}"

    return image_url

# Load the TFLite model
model_path = "detection/model.tflite"
labels_path = "detection/labels.txt"
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Load class names
with open(labels_path, "r") as file:
    class_names = [line.strip() for line in file.readlines()]

def home(request):
    return render(request, 'detection/index.html')

def upload(request):
    return render(request, 'detection/upload.html')

def weather_view(request):
    return render(request, 'detection/weather.html')

def upload_image(request):

    if request.method == 'POST':
        upload_form = ImageUploadForm(request.POST, request.FILES)

        if upload_form.is_valid():

            image_file = request.FILES['image']

            # Upload image to Azure Blob Storage
            image_url = upload_to_blob(image_file)

            # Run prediction
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
            'error': 'Invalid form submission. Please try again.'
        })

    upload_form = ImageUploadForm()

    return render(request, 'detection/upload_image.html', {
        'upload_form': upload_form
    })

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
            else:
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