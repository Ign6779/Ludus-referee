import pandas as pd
from PIL import Image, ImageOps, ImageEnhance
import os

# Load the CSV without setting column names
df = pd.read_csv(r"C:\Users\ignac\Documents\InHolland\Year 3\Ludus project\Ludus-referee\siamese\image_pairs2.csv", header=None)

# Define the directory where the images are stored
image_directory = r"C:\Users\ignac\Documents\InHolland\Year 3\Ludus project\Ludus-referee\siamese\image_dataset2"

def augment_image(image_path):
    """Apply various augmentations and save new images."""
    try:
        img = Image.open(image_path)
        if img.size != (1920, 1080):
            img = img.resize((1920, 1080))
        
        augmentations = {
            'flipped': ImageOps.mirror(img),
            'brightness': ImageEnhance.Brightness(img).enhance(1.5),
            'contrast': ImageEnhance.Contrast(img).enhance(1.5)
        }
        
        augmented_paths = []
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        
        for key, aug_image in augmentations.items():
            new_name = f"{base_name}_{key}.jpg"
            new_path = os.path.join(image_directory, new_name)
            aug_image.save(new_path)
            augmented_paths.append(new_name)
        
        return augmented_paths

    except IOError as e:
        print(f"Error processing image {image_path}: {e}")
        return []

new_rows = []

for _, row in df.iterrows():
    # Generate augmented images for each image in the pair, using index to access columns
    img1_aug_paths = augment_image(os.path.join(image_directory, row[0]))
    img2_aug_paths = augment_image(os.path.join(image_directory, row[1]))
    
    # Combine augmented images into new pairs with the same label
    for img1_path in img1_aug_paths:
        for img2_path in img2_aug_paths:
            new_rows.append([img1_path, img2_path, row[2]])

# Add augmented pairs to the original DataFrame
augmented_df = pd.DataFrame(new_rows, columns=[0, 1, 2])
final_df = pd.concat([df, augmented_df])

# Save the updated DataFrame to CSV with a specific path
try:
    final_df.to_csv(r"C:\Users\ignac\Documents\InHolland\Year 3\Ludus project\Ludus-referee\siamese\augmented_image_pairs.csv", index=False, header=False)
    print("CSV file saved successfully.")
except Exception as e:
    print(f"Error saving the CSV file: {e}")
