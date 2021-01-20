import base64
import numpy as np
import io
from PIL import Image
import keras
from keras import backend as K
from keras.models import Sequential
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from flask import request
from flask import jsonify
from flask import Flask
import pickle
import tensorflow as tf


model = tf.keras.models.load_model("modelsav")


def preprocess_image(image, target_size):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    return image


def predict():

    #encoded = message['image']
    encoded = my_string
    decoded = base64.b64decode(encoded)
    image = Image.open(io.BytesIO(decoded))
    processed_image = preprocess_image(image, target_size=(128, 128))

    prediction = model.predict(processed_image)
    print(prediction)
    preclass = {0: "Apple_healthy", 1: "Apple_unhealthy", 2: "Pepper_bell_healthy", 3: "Pepper_bell_bacterial_spot", 4: "Cherry_(includingsor)_healthy", 5: "Cherry_(includingsor)_Powdery_mildew", 6: "Corn_(Maize)_healthy", 7: "Corn_(Maize)_unhealthy",
                8: "Grape_healthy", 9: "Grape_unhealthy", 10: "Peach_healthy", 11: "Peach_bacterial_spot", 12: "Potato_healthy", 13: "Potato_unhealthy", 14: "Strawberry_healthy", 15: "Strawberry_Leaf_scorch", 16: "Tamato_healthy", 17: "Tamato_unhealthy"}
    response = np.argmax(prediction)
    return response


execute = predict()
print(execute)
