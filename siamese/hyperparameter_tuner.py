import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from kerastuner.tuners import Hyperband
import siamese_networks as snet
import pair_creator as pcre
from sklearn.model_selection import train_test_split

def build_model(hp):
    input_shape = (1920, 1080, 1)  # Fixed input size as per your requirements
    input_a = Input(shape=input_shape, name='left_input')
    input_b = Input(shape=input_shape, name='right_input')

    # The base network architecture with hyperparameter tunable options
    x = Conv2D(hp.Int('num_filters_1', min_value=32, max_value=128, step=32), 
               (hp.Choice('kernel_size_1', values=[3, 5, 11]), hp.Choice('kernel_size_1', values=[3, 5, 11])), 
               strides=(4, 4), padding='same', activation='relu')(input_a)
    x = MaxPooling2D(pool_size=(4, 4))(x)

    x = Conv2D(hp.Int('num_filters_2', min_value=64, max_value=256, step=64), 
               (hp.Choice('kernel_size_2', values=[3, 5, 7]), hp.Choice('kernel_size_2', values=[3, 5, 7])), 
               activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(4, 4))(x)

    x = Conv2D(hp.Int('num_filters_3', min_value=128, max_value=512, step=128), 
               (hp.Choice('kernel_size_3', values=[3, 5]), hp.Choice('kernel_size_3', values=[3, 5])), 
               activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(hp.Int('num_filters_4', min_value=256, max_value=512, step=128), 
               (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(hp.Int('num_filters_5', min_value=256, max_value=512, step=128), 
               (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Flatten()(x)
    x = Dense(hp.Int('dense_units_1', min_value=2048, max_value=4096, step=1024), activation='relu')(x)
    x = Dropout(hp.Float('dropout_1', min_value=0, max_value=0.5, step=0.1))(x)
    x = Dense(hp.Int('dense_units_2', min_value=1024, max_value=2048, step=512), activation='relu')(x)
    x = Dropout(hp.Float('dropout_2', min_value=0, max_value=0.5, step=0.1))(x)
    x = Dense(1024, activation='relu')(x)

    base_network = Model(inputs=input_a, outputs=x)

    processed_a = base_network(input_a)
    processed_b = base_network(input_b)

    distance = Lambda(snet.euclidean_distance, output_shape=snet.euclidean_distance_output_shape)([processed_a, processed_b])
    model = Model(inputs=[input_a, input_b], outputs=distance)

    model.compile(loss=snet.contrastive_loss,
                  optimizer=Adam(learning_rate=hp.Float('lr', min_value=1e-5, max_value=1e-2, sampling='LOG')),
                  metrics=[lambda y_true, y_pred: snet.binary_accuracy(y_true, y_pred, hp.Float('threshold', min_value=0.01, max_value=0.2, step=0.01))])

    return model

# Hyperband tuner configuration
tuner = Hyperband(
    build_model,
    objective='val_loss',
    max_epochs=10,
    factor=3,
    directory='siamese_model_tuning',
    project_name='siamese_network_hyperband'
)

# Assuming 'images_folder' is the path to your images folder and 'csv_path' is the path to your CSV file
pairs, labels = pcre.prepare_data(r"C:\Users\ignac\Documents\InHolland\Year 3\Ludus project\Ludus-referee\siamese\image_pairs.csv", r"C:\Users\ignac\Documents\InHolland\Year 3\Ludus project\Ludus-referee\siamese\image_dataset")

# Split pairs and labels for training, if necessary. Here pairs[:, 0] and pairs[:, 1] are used directly.

x_train, x_test, y_train, y_test = train_test_split(pairs, labels, test_size=0.20, random_state=42)

# Perform the hyperparameter search
tuner.search(x_train[:, 0], x_train[:, 1], y_train, epochs=3, validation_split=0.2)  # Modify epochs, validation_split as needed

# Get the optimal hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print(f"The optimal number of filters in the first layer is {best_hps.get('num_filters_1')}")
print(f"The optimal learning rate for the optimizer is {best_hps.get('lr')}")
print(f"The optimal threshold for binary accuracy calculation is {best_hps.get('threshold')}")

# Retrieve the best model
best_model = tuner.get_best_models(num_models=1)[0]