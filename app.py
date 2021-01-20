import base64
import numpy as np
import io
from PIL import Image
from flask import request
from flask import jsonify
from flask import Flask
import pickle
import tensorflow as tf

app = Flask(__name__)
#model = tf.keras.models.load_model("modelsav")


def preprocess_image(image, target_size):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = np.expand_dims(image, axis=0)
    return image


def load_model():
    # load the pre-trained Keras model (here we are using a model
    # pre-trained on ImageNet and provided by Keras, but you can
    # substitute in your own networks just as easily)
    global model
    model = pickle.load(open("leaf_finalized_model.sav", 'rb'))
    model.compile(optimizer='adam', loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


@ app.route("/")
def index():
    return "Api is working go to /api/predict to get predction with img as input"


@ app.route('/api/predict', methods=['POST'])
def predict():
    message = request.get_json(force=True)
    encoded = message['image']
    decoded = base64.b64decode(encoded)
    image = Image.open(io.BytesIO(decoded))
    processed_image = preprocess_image(image, target_size=(128, 128))
    prediction = model.predict(processed_image)
    print(prediction)
    preclass = {0: "Apple_healthy", 1: "Apple_unhealthy", 2: "Pepper_bell_healthy", 3: "Pepper_bell_bacterial_spot", 4: "Cherry_(includingsor)_healthy", 5: "Cherry_(includingsor)_Powdery_mildew", 6: "Corn_(Maize)_healthy", 7: "Corn_(Maize)_unhealthy",
                8: "Grape_healthy", 9: "Grape_unhealthy", 10: "Peach_healthy", 11: "Peach_bacterial_spot", 12: "Potato_healthy", 13: "Potato_unhealthy", 14: "Strawberry_healthy", 15: "Strawberry_Leaf_scorch", 16: "Tamato_healthy", 17: "Tamato_unhealthy"}
    response = preclass[np.argmax(prediction)]
    return jsonify(response)


if __name__ == '__main__':
    # port = int(os.environ.get('PORT', 5000))
    print(("* Loading Keras model and Flask starting server..."
           "please wait until server has fully started"))
    load_model()
    app.run()
