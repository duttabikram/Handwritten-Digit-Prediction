import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt


# 1. Load MNIST dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# 2. Normalize pixel values
# Original pixels: 0-255
# Convert them to: 0-1
x_train = x_train / 255.0
x_test = x_test / 255.0

# 3. Add channel dimension
# CNN expects: (images, height, width, channels)
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]


# 4. Create CNN model
model = keras.Sequential([
    
    layers.Conv2D(
        32,
        kernel_size=(3, 3),
        activation="relu",
        input_shape=(28, 28, 1)
    ),

    layers.MaxPooling2D(pool_size=(2, 2)),

    layers.Conv2D(
        64,
        kernel_size=(3, 3),
        activation="relu"
    ),

    layers.MaxPooling2D(pool_size=(2, 2)),

    layers.Flatten(),

    layers.Dense(128, activation="relu"),

    layers.Dense(10, activation="softmax")
])


# 5. Compile the model
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# 6. Train the CNN
model.fit(
    x_train,
    y_train,
    epochs=5,
    batch_size=64,
    validation_split=0.1
)

model.save("digit_cnn.keras")

# 7. Test the model
test_loss, test_accuracy = model.evaluate(
    x_test,
    y_test
)

print("Test Accuracy:", test_accuracy)


# 8. Make a prediction
image = x_test[0]

prediction = model.predict(
    image.reshape(1, 28, 28, 1)
)

predicted_digit = prediction.argmax()

print("Actual digit:", y_test[0])
print("Predicted digit:", predicted_digit)


# 9. Display the image
plt.imshow(
    image.squeeze(),
    cmap="gray"
)

plt.title(f"Prediction: {predicted_digit}")
plt.axis("off")
plt.show()