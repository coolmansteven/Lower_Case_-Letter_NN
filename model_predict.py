# Code implemented from NNFS book
import numpy as np
import nnfs
import cv2

from main import Model

nnfs.init()


# Label index to label name relation
letter_mnist_labels = {
    0: 'a',
    1: 'b',
    2: 'c',
    3: 'd',
    4: 'e',
}

# Read an image
image_data = cv2.imread('lowera.png', cv2.IMREAD_GRAYSCALE)

# Resize to the same size as Fashion MNIST images
image_data = cv2.resize(image_data, (28, 28))

# Reshape and scale pixel data
image_data = (image_data.reshape(1, -1).astype(np.float32) - 127.5) / 127.5

# Load the model
model = Model.load('letter_mnist.model')

# Predict on the image
confidences = model.predict(image_data)

# Get prediction instead of confidence levels
predictions = model.output_layer_activation.predictions(confidences)

# Get label name from label index
prediction = letter_mnist_labels[predictions[0]]

print('Predicted Letter:', prediction)

