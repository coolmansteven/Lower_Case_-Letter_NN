import numpy as np
import cv2
import os
import nnfs

from main import Model, Layer_Dense, Activation_ReLU, Activation_Softmax, Loss_CategoricalCrossentropy, Optimizer_Adam, \
    Accuracy_Categorical

nnfs.init()

# set image size to avoid shaping errors
IMAGE_SIZE = (28, 28)


# Loads a MNIST dataset
def load_mnist_dataset(dataset, path):

    # Scan all the directories and create a list of labels
    labels = os.listdir(os.path.join(path, dataset))

    # Create lists for samples and labels
    X = []
    y = []

    # For each label folder
    for label in labels:
        if label == '.DS_Store':
            continue
        # And for each image in given folder
        for file in os.listdir(os.path.join(path, dataset, label)):
            if file == '.DS_Store':
                continue
            # Read the image
            image = cv2.imread(
                        os.path.join(path, dataset, label, file),
                        cv2.IMREAD_GRAYSCALE)

            resized_image = cv2.resize(image, IMAGE_SIZE, interpolation=cv2.INTER_LINEAR)

            # And append it and a label to the lists
            X.append(resized_image)
            y.append(label)

    # Convert the data to proper numpy arrays and return
    return np.array(X), np.array(y).astype('uint8')


# MNIST dataset (train + test)
def create_data_mnist(path):

    # Load both sets separately
    X, y = load_mnist_dataset('train', path)
    X_test, y_test = load_mnist_dataset('test', path)

    # And return all the data
    return X, y, X_test, y_test


# Create dataset
X, y, X_test, y_test = create_data_mnist('letter_mnist_images')

keys = np.array(range(X.shape[0]))
np.random.shuffle(keys)
X = X[keys]
y = y[keys]

X = (X.reshape(X.shape[0], -1).astype(np.float32) - 127.5) / 127.5
X_test = (X_test.reshape(X_test.shape[0], -1).astype(np.float32) - 127.5) / 127.5


# Instantiate the model
model = Model()
# Add layers
model.add(Layer_Dense(X.shape[1], 64))
model.add(Activation_ReLU())
model.add(Layer_Dense(64, 64))
model.add(Activation_ReLU())
model.add(Layer_Dense(64, 5))
model.add(Activation_Softmax())

model.set(
    loss=Loss_CategoricalCrossentropy(),
    optimizer=Optimizer_Adam(decay=5e-5),
    accuracy=Accuracy_Categorical()
)

# Finalize the model
model.finalize()
# Train the model
model.train(X, y, validation_data=(X_test, y_test), epochs=12, batch_size=32, print_every=100)

model.evaluate(X_test, y_test)

model.save('letter_mnist.model')





