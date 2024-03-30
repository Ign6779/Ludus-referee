import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensoflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K


#basic CNN model
def init_base_network(input_shape):
    input = Input(shape=input_shape, name="base_input")
    
    #the layers of the network. might have to tweak this stuff
    x = Conv2D(64, (7, 7), activation='relu')(input)
    x = MaxPooling2D(pool_size(2, 2))(x)

    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size(2, 2))(x)

    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size(2, 2))(x)

    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
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