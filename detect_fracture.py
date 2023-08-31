import tensorflow as tf
import cv2
import numpy as np

# Load the trained model
model = tf.keras.models.load_model('fracture_detection_model.h5')

# Load and preprocess the new x-ray image
image_path = 'new_x_ray_image.jpg'
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (150, 150))
img = np.reshape(img, (1, 150, 150, 1))
img = img / 255.

# Use the model to make a prediction
prediction = model.predict(img)

# Print the prediction
if prediction < 0.5:
    print("No fracture detected.")
else:
    print("Fracture detected.")
