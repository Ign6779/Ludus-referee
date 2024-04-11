import os
import csv

# Path to the folder containing the images
images_folder_path = r"C:\Users\ignac\Documents\InHolland\Year 3\Ludus project\Ludus-referee\siamese\image_dataset"

# The path for the CSV file we will create
output_csv_path = r"C:\Users\ignac\Documents\InHolland\Year 3\Ludus project\Ludus-referee\siamese\image_pairs.csv"


image_filenames = os.listdir(images_folder_path)

# Open the CSV file in write mode
with open(output_csv_path, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    
    # Iterate over the filenames two at a time (step of 2)
    for i in range(0, len(image_filenames), 2):
        # Write the pair of filenames to the CSV, with an extra comma at the end
        csvwriter.writerow([image_filenames[i], image_filenames[i + 1], ''])

print(f"CSV file has been created at: {output_csv_path}")
