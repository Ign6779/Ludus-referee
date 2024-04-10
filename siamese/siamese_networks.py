import numpy as np
import tensorflow as tf
import cv2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K


#basic CNN model
def init_base_network(input_shape):
    input = Input(shape=input_shape, name="base_input")
    
    #the layers of the network. might have to tweak this stuff
    x = Conv2D(64, (7, 7), activation='relu')(input)
    x = MaxPooling2D(pool_size = (2, 2))(x)

    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size = (2, 2))(x)

    x = Conv2D(256, (3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size = (2, 2))(x)

    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    return Model(inputs = input, outputs = x)


#this is how siamese networks compare
def euclidean_distance(vectors):
    #calculates the Euclidean distance between two vectors

    vector1, vector2 = vectors
    sum_square = K.sum(K.square(vector1 - vector2), axis = 1, keepdims = True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))

def euclidean_distance_output_shape(shapes):
    shape1, shape2 = shapes
    return(shape1[0], 1)


#here's the actual Siamese network
input_shape = (105, 105, 1) #have to change this depending on the dataset

base_network = init_base_network(input_shape)

input_a = Input(shape = input_shape, name = 'left_input')
input_b = Input(shape = input_shape, name = 'right_input')

#we use the exact ssame base network
processed_a = base_network(input_a)
processed_b = base_network(input_b)

#now we calculate the Euclidean distance between the two feature outputs
distance = Lambda(euclidean_distance, output_shape = euclidean_distance_output_shape)([processed_a, processed_b])


#and here's the model
model = Model(inputs = [input_a, input_b], outputs = distance)

#compile model and train
def train_model(pairs, labels, epochs = 10, batch_size = 32):
    model.compile(loss = 'binary_crossentropy', optimizer = Adam(0.0001), metrics = ['accuracy'])
    model.fit([pairs[:, 0], pairs[:, 1]], labels, epochs = epochs, batch_size = batch_size)

def test_model(pairs, labels):
    loss, accuracy = model.evaluate([pairs[:, 0], pairs[:, 1]], labels)
    print(f"Test loss: {loss}, test accuracy: {accuracy}")
    return loss, accuracy


#to actually use the model
def analyze_video(reference_image_path, video_path, threshold = 0.05):
    #load reference and process it
    reference_image = cv2.imread(reference_image, cv2.IMREAD_GRAYSCALE)
    reference_image = cv2.resize(reference_image, (105, 105))
    
    cap = cv2.VideoCapture(video_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_processed = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_processed = cv2.resize(frame_processed, (105, 105))

        distance = model.predict([np.expand_dims(reference_image, axis = 0), np.expand_dims(frame_processed, axis = 0)])

        if distance >threshold:
            return False
    
    cap.release()
    return True