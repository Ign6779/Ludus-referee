import numpy as np
import pandas as pd
import cv2
import siamese_networks as siamnet
from sklearn.model_selection import train_test_split

def load_and_preprocess_image(image_path, target_size=(1920, 1080)):
    # Load the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # Resize the image
    image = cv2.resize(image, target_size)
    # Expand dimensions to add the channel information
    image = np.expand_dims(image, axis=-1)
    # Normalize pixel values to [0, 1]
    image = image / 255.0
    return image

def prepare_data(csv_path, images_folder):
    # Load the CSV file
    df = pd.read_csv(csv_path, header=None, names=['image1', 'image2', 'label'])
    
    # Initialize lists to hold our pairs and labels
    pairs = []
    labels = []
    
    # Iterate over the DataFrame
    for index, row in df.iterrows():
        # Load and preprocess each image in the pair
        img1_path = f"{images_folder}/{row['image1']}"
        img2_path = f"{images_folder}/{row['image2']}"
        image1 = load_and_preprocess_image(img1_path)
        image2 = load_and_preprocess_image(img2_path)
        
        # Add the pair and label to their respective lists
        pairs.append([image1, image2])
        labels.append(row['label'])
        
    # Convert lists to numpy arrays
    pairs = np.array(pairs, dtype='float32')
    labels = np.array(labels, dtype='float32')
    
    # Reshape for the model's expected input
    pairs = pairs.transpose((0, 2, 3, 1, 4)).reshape((-1, 1080, 1920, 1))
    
    return pairs, labels

# Assuming 'images_folder' is the path to your images folder and 'csv_path' is the path to your CSV file
pairs, labels = prepare_data(r"C:\Users\ignac\Documents\InHolland\Year 3\Ludus project\Ludus-referee\siamese\image_pairs.csv", r"C:\Users\ignac\Documents\InHolland\Year 3\Ludus project\Ludus-referee\siamese\image_dataset")

# Split pairs and labels for training, if necessary. Here pairs[:, 0] and pairs[:, 1] are used directly.

x_train, x_test, y_train, y_test = train_test_split(pairs, labels, test_size=0.20, random_state=42)
# You might want to split your dataset into a training and validation set.

# Now, train your model
print("model training")
print()
siamnet.train_model([x_train[:, 0], x_train[:, 1]], y_train, epochs=10, batch_size=32)
print("model has been trained")
print()
print("model testing")
siamnet.test_model([x_test[:, 0], x_test[:, 1]], y_test)