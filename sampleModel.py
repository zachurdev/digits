import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Assuming x_train and y_train are defined or imported properly
# Replace these placeholder values with your actual training data
x_train = np.random.rand(1000, 28, 28)  # Placeholder for training images
y_train = np.random.randint(0, 10, size=1000)  # Placeholder for training labels

# Generate some sample test data
num_samples = 10000  # Number of samples for testing
x_test = np.random.rand(num_samples, 28, 28)  # Generating random images for testing
y_test = np.random.randint(0, 10, size=num_samples)  # Generating random labels for testing (assuming 10 classes)

# Normalize the test data (assuming pixel values range from 0 to 255)
x_test = x_test / 255.0

# Ensure labels are integers
y_test = y_test.astype(np.int32)

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])

# Define and compile the model
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model with visualization
history = model.fit(x_train, y_train, epochs=50, validation_data=(x_test, y_test))

# Visualize training progress (loss and accuracy)
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Visualize confusion matrix
y_pred = np.argmax(model.predict(x_test), axis=-1)
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# Visualize sample predictions
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.imshow(x_test[i], cmap='gray')
    plt.title(f'Predicted: {y_pred[i]}, Actual: {y_test[i]}')
    plt.axis('off')
plt.show()


# Evaluate the model on test data
model.evaluate(x_test, y_test)
