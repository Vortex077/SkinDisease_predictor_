from flask import Flask, render_template, request, redirect
import os
import keras
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
model = load_model('Dermno_RenseNet.h5')
model.make_predict_function()
train_datagen = ImageDataGenerator(zoom_range=0.5, shear_range=0.3, horizontal_flip=True, preprocessing_function=preprocess_input)
train = train_datagen.flow_from_directory(directory="dermno_copy/train", target_size=(256, 256), batch_size=32)
class_labels = list(train.class_indices.keys())

def prediction(image_path):
    try:
        img = load_img(image_path, target_size=(256, 256))
        i = img_to_array(img)
        im = preprocess_input(i)
        img = np.expand_dims(im, axis=0)
        prediction_probs = model.predict(img)[0]

        # Get the predicted class and its probability
        predicted_class = np.argmax(prediction_probs)
        confi_percentage = prediction_probs[predicted_class] * 100
        round_confi_percentage = "{:.2f}".format(confi_percentage)
        return { 'class': class_labels[predicted_class], 'confidence': round_confi_percentage}
    except Exception as e:
        print(f"Error in prediction: {e}")
        return {'class': 'Error', 'confidence': 0.0}

@app.route('/')
def index():
    return render_template('home.html')  # Replace 'your_html_filename.html' with your actual HTML file name

@app.route('/diagnose')
def diagnose():
    return render_template('diagnose.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/list')
def list():
    return render_template('list.html')

@app.route('/actinic')
def actinic():
    return render_template('actinic.html')

@app.route('/atopic')
def atopic():
    return render_template('Atopic.html')

@app.route('/benign')
def benign():
    return render_template('Benign.html')

@app.route('/dermat')
def dermat():
    return render_template('Dermat.html')

@app.route('/nevus')
def nevus():
    return render_template('Nevus.html')

@app.route('/melanoma')
def melanoma():
    return render_template('Melanoma.html')

@app.route('/squamous')
def squamous():
    return render_template('squamous.html')

@app.route('/tinea')
def tinea():
    return render_template('Tinea.html')

@app.route('/lesions')
def lesions():
    return render_template('Lesions.html')

@app.route('/submit', methods=['GET','POST'])
def submit():
    if request.method == "POST":
        image=request.files["my_image"]
        image_path = "static/"+image.filename
        image.save(image_path)
        result = prediction(image_path)
    try:
        return render_template('diagnose.html', result1=result, image_path=image_path)
    except Exception as e:
        print(f"Error in prediction: {e}")


if __name__ == '__main__':
    app.run(debug=True)
