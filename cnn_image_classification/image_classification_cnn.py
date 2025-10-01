# Import Libraries
import warnings 
warnings.filterwarnings('ignore')
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping

# For MobileNetV2
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

# --------------------
# Data Preparation
# --------------------
(X_train, y_train),(X_test,y_test) = cifar10.load_data()
X_train, X_valid,y_train,y_valid = train_test_split(X_train,y_train,test_size=0.1, random_state=0)

print('Train:', X_train.shape, y_train.shape)
print('Valid:', X_valid.shape, y_valid.shape)
print('Test :', X_test.shape, y_test.shape)

def cifar_view(X_train,y_train,class_names): 
    # Create a new figure 
    plt.figure(figsize=(15,15)) 
    # Loop over the first 25 images 
    for i in range(64): 
    # Create a subplot for each image 
       plt.subplot(8, 8, i+1) 
       plt.xticks([]) 
       plt.yticks([]) 
       plt.grid(False) 
       # Display the image 
       plt.imshow(X_train[i]) 
       # Set the label as the title 
       plt.title(class_names[y_train[i][0]], fontsize=12) 
       # Display the figure 
    plt.show() 
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'] 
cifar_view(X_train,y_train,class_names)

# Normalization
X_train = np.array([cv2.resize(img, (96, 96)) for img in X_train])
X_valid = np.array([cv2.resize(img, (96, 96)) for img in X_valid])
X_test  = np.array([cv2.resize(img, (96, 96)) for img in X_test])

mean = np.mean(X_train)
std  = np.std(X_train)

X_train = (X_train-mean)/(std+1e-7)
X_test  = (X_test-mean)/(std+1e-7)
X_valid = (X_valid-mean)/(std+1e-7)

# One-hot encoding
y_train = to_categorical(y_train, 10)
y_valid = to_categorical(y_valid, 10)
y_test  = to_categorical(y_test, 10)

# --------------------
# Data Augmentation
# --------------------
data_generator = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.12,
    height_shift_range=0.12,
    horizontal_flip=True,
    zoom_range=0.1,
    brightness_range=[0.9,1.1],
    shear_range=10,
    channel_shift_range=0.1,
)

# --------------------
# Model: MobileNetV2
# --------------------
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(96, 96, 3)
)
base_model.trainable = False  # freeze backbone

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(10, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# --------------------
# Compile
# --------------------
optimizer = Adam(learning_rate=0.0005)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# --------------------
# Callbacks
# --------------------
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001)
early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1)

# --------------------
# Train
# --------------------
history = model.fit(
    data_generator.flow(X_train, y_train, batch_size=64),
    epochs=50,   # start smaller
    validation_data=(X_valid, y_valid),
    callbacks=[reduce_lr, early_stopping],
    verbose=2
)

# --------------------
# Plot Results
# --------------------
plt.figure(figsize=(15,6))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss', color='#8502d1')
plt.plot(history.history['val_loss'], label='Validation Loss', color='darkorange')
plt.legend()
plt.title('Loss Evolution')

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy', color='#8502d1')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='darkorange')
plt.legend()
plt.title('Accuracy Evolution')

plt.show()

# --------------------
# Evaluate on Test Data
# --------------------
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=1)
print('\nTest Accuracy:', test_acc)
print('Test Loss:    ', test_loss)


    
# --------------------
# SAVE MODEL
# --------------------
model.save("cifar10_mobilenetv2.h5")
print("Model saved as cifar10_mobilenetv2.h5")

# --------------------
# LOAD MODEL (later or in a new script)
# --------------------
loaded_model = load_model("cifar10_mobilenetv2.h5")

# --------------------
# EVALUATE ON TEST DATA
# --------------------
test_loss, test_acc = loaded_model.evaluate(X_test, y_test, verbose=1)
print("\nTest Accuracy:", test_acc)
print("Test Loss:    ", test_loss)

# --------------------
# MAKE PREDICTIONS
# --------------------
predictions = loaded_model.predict(X_test)
predicted_classes = np.argmax(predictions, axis=1)

# Show some predictions
import random
for i in random.sample(range(len(X_test)), 5):
    plt.imshow((X_test[i]*std + mean).astype(np.uint8))  # denormalize for display
    plt.title(f"Predicted: {predicted_classes[i]}")
    plt.show()
