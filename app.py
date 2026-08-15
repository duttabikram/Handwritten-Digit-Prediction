import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas


# Load trained CNN
model = tf.keras.models.load_model("digit_cnn.keras")


st.title("🧠 AI Handwritten Digit Recognizer")

st.write("Draw a digit from 0 to 9 below and let the CNN recognize it!")


# Drawing canvas
canvas = st_canvas(
    fill_color="black",
    stroke_width=15,
    stroke_color="white",
    background_color="black",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas"
)


# Predict button
if st.button("🔮 Predict"):

    if canvas.image_data is not None:

        # Get canvas image
        img = canvas.image_data

        # Convert RGBA → grayscale
        img = Image.fromarray(
            img.astype("uint8")
        ).convert("L")

        # Resize to MNIST size
        img = img.resize((28, 28))

        # Convert to numpy
        img = np.array(img)

        # Normalize
        img = img / 255.0

        # Add batch + channel dimensions
        img = img.reshape(1, 28, 28, 1)

        # CNN prediction
        prediction = model.predict(
            img,
            verbose=0
        )

        # Get predicted digit
        digit = np.argmax(prediction)

        # Confidence
        confidence = np.max(prediction) * 100

        st.success(
            f"Prediction: **{digit}**"
        )

        st.metric(
            "Confidence",
            f"{confidence:.2f}%"
        )


        # Show probabilities
        st.subheader("Prediction probabilities")

        probabilities = prediction[0]

        for i, probability in enumerate(probabilities):

            st.write(
                f"**{i}** — {probability * 100:.2f}%"
            )

            st.progress(
                float(probability)
            )