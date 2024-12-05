import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import os
import sys
import random

# Load the submission file from command-line arguments
if len(sys.argv) > 1:
    submission_file = sys.argv[1]
else:
    print("Please provide the path to the CSV file.")
    sys.exit(1)

df = pd.read_csv(submission_file)

# Determine the image folder based on the CSV file name
image_folder = 'train' if os.path.basename(submission_file) == 'Train.csv' else 'test'

# Get image IDs from command-line arguments or select random images
if len(sys.argv) > 2:
    image_ids = sys.argv[2:5]
else:
    image_ids = random.sample(list(df['Image_ID'].unique()), 3)

# Set up the plot
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Iterate over the selected images      
for ax, image_id in zip(axes.flatten(), image_ids):
    # Load the image
    image_path = os.path.join(f'/home/guillermo/Documents/AI2/challenge-malaria/{image_folder}', image_id)
    image = Image.open(image_path)
    ax.imshow(image)
    ax.set_title(image_id)
    
    # Draw bounding boxes and class labels
    image_data = df[df['Image_ID'] == image_id]
    for _, row in image_data.iterrows():
        ymin, xmin, ymax, xmax = row['ymin'], row['xmin'], row['ymax'], row['xmax']
        width, height = xmax - xmin, ymax - ymin
        rect = patches.Rectangle((xmin, ymin), width, height, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        # Add class label
        class_label = 'TPZ' if row['class'] == 'Trophozoite' else row['class']
        ax.text(xmin, ymin - 10, class_label, color='r', fontsize=12, weight='bold')

# Adjust layout and show the plot
plt.tight_layout()
plt.show()
