# Load MNIST dataset
import tensorflow as tf


MNIST = tf.keras.datasets.mnist

# Split dataset into training and testing sets
(x_train, y_train), (x_test, y_test) = MNIST.load_data()

# Normalize the data
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)


# Load the model
trainedModel = tf.keras.models.load_model('trainedNumber.model')

# Test the model
loss, accuracy = trainedModel.evaluate(x_test, y_test)
print('Loss: ', loss)
print('Accuracy: ', accuracy)
