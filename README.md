# 🏀 Basketball Legend Classifier

A web app that identifies basketball legends from images using a deep learning model. Upload an image of Kobe Bryant, LeBron James, Michael Jordan, Shaquille O'Neal, or Stephen Curry, and the app predicts who it is!

---

## Demo

https://github.com/user-attachments/assets/b97af647-bf5b-4795-982f-41eb0c6b4f7c

---

## Tech Stack

- **Frontend & UI:** [Streamlit](https://streamlit.io/)
- **Model Framework:** TensorFlow, Keras
- **Python Libraries:**
  - tensorflow
  - numpy
  - Pillow
  - base64
  - io
- **Model:** Custom CNN trained on basketball player images
- **Training Environment:** Google Colab (GPU)
- **Deployment:** Render

---

## Folder Structure

```text
image-classifier/
│
├── basketball_dataset/
│   ├── kobe_bryant/
│   ├── lebron_james/
│   ├── michael_jordan/
│   ├── shaquille_oneal/
│   └── stephen_curry/
│
├── basketball-classifier-app/
│   ├── app.py
│   ├── model.h5
│   ├── requirements.txt
│   ├── images/
│   ├── sample_images/
│   └── ...
│
├── README.md
├── requirements.txt
└── scrape_images.py
```

---

# Model Training

Google Colab was used for training because it provides free access to GPUs, significantly reducing training time without requiring powerful local hardware.

### Dataset

Images were collected for five basketball legends:

- Kobe Bryant
- LeBron James
- Michael Jordan
- Shaquille O'Neal
- Stephen Curry

```python
from google.colab import files
uploaded = files.upload()
```

```python
import zipfile
import os

with zipfile.ZipFile("basketball_dataset.zip", "r") as zip_ref:
    zip_ref.extractall("basketball_dataset")

os.listdir("basketball_dataset")
```

### Preprocessing

- Images resized to **100×100**
- Pixel values normalized between **0 and 1**

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_gen = datagen.flow_from_directory(
    "basketball_dataset",
    target_size=(100,100),
    batch_size=16,
    class_mode="categorical",
    subset="training"
)

val_gen = datagen.flow_from_directory(
    "basketball_dataset",
    target_size=(100,100),
    batch_size=16,
    class_mode="categorical",
    subset="validation"
)
```

### Model Architecture

A custom Convolutional Neural Network (CNN) built using TensorFlow/Keras.

### Training

- Adam Optimizer
- Categorical Crossentropy Loss
- Multiple epochs until validation accuracy stabilized

```python
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=10
)
```

### Save Model

```python
model.save("model.h5")
```

```python
from google.colab import files
files.download("model.h5")
```

---

# How to Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/basketball-classifier.git
```

### 2. Navigate into the app

```bash
cd basketball-classifier-app
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
streamlit run app.py
```

Upload your own image or choose one of the sample images.

---

# Features

- Upload your own image
- Choose from sample images
- Glassmorphism UI
- Confidence scores
- Real-time predictions
- Background image
- Responsive layout

---

# Challenges Faced

1. Collecting enough quality images for each player.

2. Dealing with different camera angles, lighting conditions, and backgrounds.

3. Preventing overfitting while maintaining good validation accuracy.

4. Building a clean and responsive Streamlit interface with custom CSS.

5. Managing image uploads, sample image selection, and prediction flow using Streamlit Session State.

---

# Future Improvements

- Support more NBA players.

- Increase model accuracy using transfer learning.

- Add Grad-CAM visualizations to explain predictions.

- Improve mobile responsiveness.

- Deploy on Streamlit Community Cloud.

---

# Author

**Developed by Ajey**



