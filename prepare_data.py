import os
import random
import shutil

# Define paths to your train and validation directories
train_dir = 'data/train'
validation_dir = 'data/validation'

# Create the validation directory if it doesn't exist
os.makedirs(validation_dir, exist_ok=True)

# Get the list of class directories in the train directory
class_directories = [f.name for f in os.scandir(train_dir) if f.is_dir()]

# Define the percentage of images to move to validation
validation_percentage = 0.1

# Move images from train to validation directories
for class_dir in class_directories:
    class_train_dir = os.path.join(train_dir, class_dir)
    class_validation_dir = os.path.join(validation_dir, class_dir)

    os.makedirs(class_validation_dir, exist_ok=True)

    # List all files in the class directory
    class_files = [f.name for f in os.scandir(class_train_dir) if f.is_file()]

    # Calculate the number of images to move
    num_validation = int(len(class_files) * validation_percentage)

    # Randomly select images to move
    validation_files = random.sample(class_files, num_validation)

    # Move the selected images to the validation directory
    for file in validation_files:
        src_path = os.path.join(class_train_dir, file)
        dst_path = os.path.join(class_validation_dir, file)
        shutil.move(src_path, dst_path)

print("Validation subset created successfully.")
