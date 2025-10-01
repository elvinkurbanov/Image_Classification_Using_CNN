Load & Prepare Data

Loads the CIFAR-10 dataset (10 classes: airplane, car, cat, etc.).

Splits into train/validation/test sets.

Resizes images to 96×96, normalizes pixel values, and one-hot encodes labels.

Data Augmentation

Uses ImageDataGenerator to randomly rotate, shift, flip, zoom, and adjust brightness of images to make training more robust.

Model (MobileNetV2)

Loads a pretrained MobileNetV2 (on ImageNet) as a feature extractor.

Freezes its weights, then adds new layers for CIFAR-10 classification.

Compile & Train

Optimizer: Adam, loss: categorical crossentropy, metric: accuracy.

Uses callbacks (ReduceLROnPlateau, EarlyStopping) to adjust learning rate and stop overfitting.

Trains the model on augmented data.

Evaluate & Visualize

Plots training/validation accuracy and loss curves.

Evaluates final accuracy on the test set.

Save & Reload

Saves trained model as cifar10_mobilenetv2.h5.

Reloads the model to confirm it works.

Predictions

Makes predictions on test images.

Displays a few sample images with their predicted class.