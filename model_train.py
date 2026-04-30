from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Define paths to your train and validation directories
train_dir = 'data/train'
validation_dir = 'data/validation'

# Define target image dimensions
img_size = (224, 224)  # VGG16's default input size

# Create data generators for training and validation with augmentation
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

validation_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0
)

batch_size = 64

train_generator = train_datagen.flow_from_directory(
    train_dir,
    batch_size=batch_size,
    class_mode='categorical',
    target_size=img_size
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    batch_size=batch_size,
    class_mode='categorical',
    target_size=img_size
)

# Load VGG16 as the base model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(img_size[0], img_size[1], 3))

# Build your custom model on top of the base model
model = Sequential()
model.add(base_model)
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_generator.class_indices), activation='softmax'))

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
epochs = 32
model.fit(train_generator, epochs=epochs, validation_data=validation_generator)

# Save the trained model
#model.save('cancer_detection_model.h5')  ## run this line if saving model in .keras is giving any error
#model.save('cancer_detection_model.keras')  ## run this to save model in .keras format
