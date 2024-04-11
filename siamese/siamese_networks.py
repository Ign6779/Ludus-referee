import numpy as np
import tensorflow as tf
import cv2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Lambda, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K


#basic CNN model - ADAPTED FOR 1080p RESOLUTION
def init_base_network(input_shape):
    input = Input(shape=input_shape, name="base_input")
    
    #the layers of the network. might have to tweak this stuff
    x = Conv2D(64, (11, 11), strides=(4, 4), padding='same')(input)
    x = MaxPooling2D(pool_size=(4, 4))(x)

    x = Conv2D(128, (7, 7), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size = (4, 4))(x)

    x = Conv2D(256, (5, 5), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size = (2, 2))(x)

    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Flatten()(x)

    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.5)(x)

    x = Dense(2048, activation='relu')(x)
    x = Dropout(0.5)(x)

    x = Dense(1024, activation='relu')(x)

    return Model(inputs=input, outputs=x)


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
input_shape = (1920, 1080, 1) #have to change this depending on the dataset!!

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

#have to define a different loss function - contrastive loss
def contrastive_loss(y_true, y_pred):
    margin = 1
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * square_pred + (1 - y_true) * margin_square)

#compile model and train
def train_model(x_train1, x_train2, labels, epochs=10, batch_size=32):
    model.compile(loss=contrastive_loss, optimizer=Adam(0.0001))
    model.fit([x_train1, x_train2], labels, epochs=epochs, batch_size=batch_size)

def test_model(x_test1, x_test2, labels, threshold=0.5):
    predictions = model.predict([x_test1, x_test2])
    predictions = (predictions.ravel() <= threshold).astype(int)
    accuracy = np.mean(predictions == labels)
    print(f"Test accuracy: {accuracy * 100:.2f}%")
    return accuracy

#to actually use the model
def analyze_video(reference_image_path, video_path, threshold = 0.05):
    #load reference and process it
    reference_image = cv2.imread(reference_image_path, cv2.IMREAD_GRAYSCALE)
    
    cap = cv2.VideoCapture(video_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_processed = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        distance = model.predict([np.expand_dims(reference_image, axis = 0), np.expand_dims(frame_processed, axis = 0)])

        if distance > threshold:
            return False
    
    cap.release()
    return True


def analyze_image(image_path1, image_path2, threshold=0.05):
    image1 = cv2.imread(image_path1, cv2.IMREAD_GRAYSCALE)
    #image1 = cv2.resize(image1, (1920, 1080))

    image2 = cv2.imread(image_path2, cv2.IMREAD_GRAYSCALE)
    #image2 = cv2.resize(image2, (1920, 1080))

    image1 = np.expand_dims(image1, axis=0)
    image1 = np.expand_dims(image1, axis=-1)

    image2 = np.expand_dims(image2, axis=0)
    image2 = np.expand_dims(image2, axis=-1)

    distance = model.predict([image1, image2])

    if distance <= threshold:
        return True
    else:
        return False