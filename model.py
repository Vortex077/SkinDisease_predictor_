import keras
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator,img_to_array,load_img
from keras.applications.inception_resnet_v2 import InceptionResNetV2,preprocess_input
import numpy as np
import matplotlib.pyplot as plt


# Load the pre-trained model
model = load_model(r"C:\Users\AVIGHYAT\Dermno_RenseNet.h5")

train_datagen=ImageDataGenerator(zoom_range=0.5,shear_range=0.3,horizontal_flip=True,preprocessing_function=preprocess_input )
#val_datagen=ImageDataGenerator(preprocessing_function=preprocess_input)
train=train_datagen.flow_from_directory(directory=r"C:\Users\AVIGHYAT\dermno_copy\train",target_size=(256,256),batch_size=32)
class_labels = list(train.class_indices.keys())


# Define a function to make predictions
def prediction(image_path):
    try:
        img = load_img(image_path, target_size=(256, 256))
        i = img_to_array(img)
        im = preprocess_input(i)
        img = np.expand_dims(im, axis=0)
        prediction_probs = model.predict(img)[0]

        # Get the predicted class and its probability
        predicted_class = np.argmax(prediction_probs)
        confidence_percentage = prediction_probs[predicted_class] * 100
        return {'class': class_labels[predicted_class], 'confidence': confidence_percentage}
    except Exception as e:
        print(f"Error in prediction: {e}")
        return {'class': 'Error', 'confidence': 0.0}
        

# Existing code to get prediction
#predicted_class = model.predict(input_image)

# Modify to get prediction probabilities
#prediction_scores = model.predict(input_image)
#predicted_class = np.argmax(prediction_scores)

# Get the probability score for the predicted class
#probability_of_predicted_class = prediction_scores[predicted_class]

# Display the result to the user
#print(f"Predicted Disease: {class_labels[predicted_class]}")
#print(f"Probability: {probability_of_predicted_class * 100:.2f}%")
