# Cattle Breed Classifier

A deep learning-based image classification project that predicts cattle breeds from images using TensorFlow. This project includes dataset preparation, model training, and a user-friendly prediction interface built with Streamlit.

---

# Project Overview

The Cattle Breed Classifier is designed to identify cattle breeds from uploaded images using computer vision and deep learning techniques.

The project workflow includes:

* Dataset preparation and splitting
* Model training using TensorFlow
* Breed prediction through a Streamlit application

This project demonstrates practical implementation of:

* Deep Learning
* Image Classification
* TensorFlow Model Training
* Data Preprocessing
* Streamlit Deployment

---

# Tech Stack

## Machine Learning & Backend

* Python
* TensorFlow
* NumPy
* JSON

## Data Preparation

* splitfolders

## Frontend / Interface

* Streamlit

---

# Project Structure

```plaintext id="qsbj4q"
cattle_breed_classifier/
│
├── train.py
├── predict.py
├── prepare_data.py
├── dataset/
├── classnames.json
└── README.md
```

---

# Features

* Train a cattle breed classification model
* Dataset splitting and preprocessing
* Upload cattle images for prediction
* Interactive prediction interface using Streamlit
* Class label management using JSON
* Real-time breed prediction

---

# Libraries Used

## train.py

* TensorFlow
* JSON

## predict.py

* Streamlit
* NumPy
* TensorFlow
* JSON

## prepare_data.py

* splitfolders

---

# Installation & Setup

## 1. Clone the Repository

```bash id="v77drw"
git clone <your_repository_link>
cd cattle_breed_classifier
```

---

## 2. Create Virtual Environment

```bash id="m3jcdx"
python -m venv venv
```

### Activate Virtual Environment

#### Windows

```bash id="g8vj7v"
venv\Scripts\activate
```

#### Mac/Linux

```bash id="0zwgsp"
source venv/bin/activate
```

---

## 3. Install Dependencies

```bash id="mkwkj7"
pip install tensorflow streamlit numpy split-folders
```

---

# Preparing the Dataset

Run the dataset preparation script:

```bash id="mbtz52"
python prepare_data.py
```

This script splits the dataset into training, validation, and testing folders.

---

# Training the Model

Run:

```bash id="d5aq2v"
python train.py
```

This will train the TensorFlow model on the cattle breed dataset.

---

# Running the Prediction App

Start the Streamlit application:

```bash id="ud6qsr"
streamlit run predict.py
```

After running the command, open the local Streamlit URL in your browser.

---

# How It Works

1. Dataset is prepared using `prepare_data.py`
2. The model is trained using TensorFlow in `train.py`
3. Users upload cattle images through the Streamlit interface
4. The trained model predicts the cattle breed
5. Predicted class names are fetched using `classnames.json`

---

# Learning Outcomes

Through this project, I gained practical experience in:

* Deep Learning workflows
* TensorFlow model training
* Dataset preprocessing
* Image classification
* Building ML applications with Streamlit
* Managing class labels using JSON

---

# Future Improvements

* Improve model accuracy
* Add support for more cattle breeds
* Deploy the application online
* Add confidence score visualization
* Enhance UI design

---

# License

This project is created for educational and learning purposes.
