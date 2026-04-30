import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import preprocess_input

# Load the trained model
#model = load_model('cancer_detection_model.h5')  ## use this if model is saved in .h5
#model = load_model('cancer_detection_model.keras')  ## use this if model is saved in .keras format

# Define path to your test directory
test_dir = 'data/test'

# Define target image dimensions (same as used during training)
img_size = (224, 224)

# Create a data generator for test data
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    batch_size=1,  # Use batch size of 1 to process images individually
    class_mode='categorical',
    target_size=img_size,
    shuffle=False  # Keep the order of predictions consistent with filenames
)

# Get true labels and predicted labels
true_labels = test_generator.classes
num_samples = len(true_labels)

# Generate predictions for the test data
predictions = model.predict(test_generator)
predicted_labels = np.argmax(predictions, axis=1)

# Generate confusion matrix
confusion_mat = confusion_matrix(true_labels, predicted_labels)

# Plot confusion matrix as a heatmap
plt.figure(figsize=(8, 6))
plt.imshow(confusion_mat, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
plt.xticks(np.arange(len(test_generator.class_indices)), test_generator.class_indices, rotation=45)
plt.yticks(np.arange(len(test_generator.class_indices)), test_generator.class_indices)
plt.ylabel('True Label')
plt.xlabel('Predicted Label')

# Save the confusion matrix plot to a file
confusion_matrix_file = 'confusion_matrix.png'
plt.savefig(confusion_matrix_file)
plt.close()  # Close the plot to prevent display

# Print classification report
target_names = list(test_generator.class_indices.keys())
classification_rep = classification_report(true_labels, predicted_labels, target_names=target_names)

# Save the classification report to a text file
classification_report_file = 'classification_report.txt'
with open(classification_report_file, 'w') as f:
    f.write(classification_rep)

print("Confusion matrix plot saved to:", confusion_matrix_file)
print("Classification report saved to:", classification_report_file)
