# Handwritten-Digit-Prediction

**CNN Digit Recognizer — A machine learning project that uses a Convolutional Neural Network (CNN) to recognize handwritten digits from 0–9, with an interactive Streamlit interface for drawing and real-time predictions.**

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![License](https://img.shields.io/badge/license-none-lightgrey)
![Python](https://img.shields.io/badge/python-3.8+-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/tensorflow-2.x-FF6F20?logo=tensorflow)
![Streamlit](https://img.shields.io/badge/streamlit-1.x-FF4B4B?logo=streamlit)

---

## Description

**Handwritten-Digit-Prediction** is a machine learning project that leverages a **Convolutional Neural Network (CNN)** to classify handwritten digits from 0 to 9. The model is trained on the **MNIST dataset**, a widely used benchmark in the field of computer vision.

This project includes:

- A **deep learning model** built using **TensorFlow/Keras**, specifically designed for digit recognition.
- An **interactive web application** powered by **Streamlit**, which allows users to draw digits on a canvas and receive real-time predictions.
- A clean and intuitive UI that displays the predicted digit along with its confidence score.

The goal of this project is to demonstrate how CNNs can be effectively applied to image classification tasks, particularly in recognizing handwritten numerals, and to provide an accessible way for users to test the model through a visual interface.

---

## Installation

Follow these steps to set up the project locally:

### 1. Clone the repository

```bash
git clone https://github.com/duttabikram/Handwritten-Digit-Prediction.git
cd Handwritten-Digit-Prediction
```

### 2. Set up a virtual environment (optional but recommended)

```bash
python -m venv venv
source venv/bin/activate   # On Windows use: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Ensure the trained model is available

Make sure the trained model file `digit_cnn.keras` is present in the root directory. If it's not available, you can train the model using `cnn_mnist.py`.

---

## Usage

### Run the Streamlit App

To launch the interactive digit recognition app:

```bash
streamlit run app.py
```

Once the app is running:

1. Draw a digit (0–9) on the canvas using your mouse or touchpad.
2. Click the **"Predict"** button.
3. View the predicted digit and its confidence score in real time.

### Train the Model (Optional)

If you need to retrain the CNN model:

```bash
python cnn_mnist.py
```

This will train the model on the MNIST dataset and save it as `digit_cnn.keras`, which is then used by the Streamlit app for inference.

---

## Tech Stack

| Technology             | Purpose                                      |
|------------------------|----------------------------------------------|
| Python                 | Core programming language                    |
| TensorFlow             | Deep learning framework for building the CNN |
| Keras                  | High-level API for neural networks           |
| Streamlit              | Web framework for the interactive UI         |
| streamlit-drawable-canvas | Canvas component for drawing input         |
| NumPy                  | Numerical computation                        |
| Pillow (PIL)           | Image processing                             |
| Matplotlib             | Visualization (if used in training)          |

---

## Features

- ✅ Trained **Convolutional Neural Network (CNN)** for digit recognition.
- ✅ Real-time prediction from user-drawn input via **Streamlit** canvas.
- ✅ Confidence score displayed alongside prediction.
- ✅ Preprocessing pipeline: grayscale conversion, resizing, normalization.
- ✅ Simple and intuitive user interface.
- ✅ Easy-to-run scripts for both training and inference.

---

## Contributing

Contributions are welcome! Here's how you can contribute:

1. **Fork** the repository.
2. Create a new branch:

   ```bash
   git checkout -b feature/YourFeature
   ```

3. Commit your changes:

   ```bash
   git commit -m 'Add some feature'
   ```

4. Push to the branch:

   ```bash
   git push origin feature/YourFeature
   ```

5. Open a **pull request**.

Please make sure to update tests and documentation as needed.

---

## License

This project does not currently have a license. If you intend to use or distribute this code, please contact the repository owner for clarification.

---

## Acknowledgements

- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)
- [Streamlit](https://streamlit.io/)
- [TensorFlow](https://www.tensorflow.org/)
- [Streamlit Drawable Canvas](https://github.com/andfanilo/streamlit-drawable-canvas)