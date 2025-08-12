[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE) 
[![Python 3.9](https://img.shields.io/badge/Python-3.9-blue.svg)](https://www.python.org/) 
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.19-orange)](#) 
[![Jupyter](https://img.shields.io/badge/Jupyter%20Notebook-F37626.svg?logo=jupyter)](#)
# Classification of Pet Face

## 🐾 Project Description

A deep learning model that classifies dog and cat breeds from facial images using the Oxford-IIIT Pet Dataset.  
Built with TensorFlow and trained on 37 breeds, it achieves high accuracy in fine-grained breed recognition.

**Key Details:**
- **Dataset:** Oxford-IIIT Pet Dataset (7,000+ images, 37 breeds)
- **Model:** Convolutional Neural Network (CNN) in TensorFlow/Keras
- **Accuracy:** *[insert your test accuracy here]*
- **Features:**
  - Image preprocessing & augmentation
  - Breed prediction from pet face photos
  - Training & validation performance visualization

---
## Table of Contents

- [Features](#features)  
- [Installation](#installation)  
- [Usage](#usage)  
- [Results](#results)  
- [Tech Stack](#tech-stack)  
- [Contributing](#contributing)  
- [Authors](#authors)  
- [License](#license)  
- [Acknowledgements](#acknowledgements)  
- [Support](#support)

---
## ✨ Features

- **Breed Classification** – Identifies 37 cat and dog breeds from facial images.
- **Oxford-IIIT Pet Dataset** – Utilizes a high-quality dataset with 7,000+ images.
- **Deep Learning Model** – Built using a Convolutional Neural Network (CNN) with TensorFlow/Keras.
- **Image Preprocessing** – Includes resizing, normalization, and augmentation for better generalization.
- **Performance Visualization** – Training and validation accuracy/loss curves for easy evaluation.
- **Reproducibility** – Implemented in Jupyter Notebook for easy execution and modification.


---
## 🛠 Installation

To set up and run the **Classification of Pet Faces** project locally:

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/classification-of-pet-faces.git
cd classification-of-pet-faces
```
### 2. Create and activate a virtual environment (recommended)
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate
```
### 3. Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```
### 4. (Optional) Manually download the Oxford-IIIT Pet Dataset
```bash
mkdir data
cd data
wget https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz
wget https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz
tar -xvzf images.tar.gz
tar -xvzf annotations.tar.gz
cd ..
```
### 5. Run the Jupyter Notebook
```bash
jupyter notebook
# Open 'classification_of_pet_faces.ipynb' and run all cells
```

---
## 🚀 Usage / Example

### 0️⃣ Download the Pretrained Model

Before running the notebook or prediction script, download the pretrained model by running:

```bash
python scripts/download_weights.py
```
### 1️⃣ Run the Jupyter Notebook (Interactive Training & Testing)
```bash

jupyter notebook
# Open 'classification_of_pet_faces.ipynb' and run all cells
```
### 2️⃣ Predict a Single Image (Using a Trained Model)
```bash
python predict.py --image_path path/to/image.jpg
```
## Example:
```bash
python predict.py --image_path sample_images/cat_01.jpg
# Output:
# Predicted Breed: Abyssinian
# Confidence: 93.4%
```

---
## 📊 Results


**Test accuracy:** 0.87 • **Macro F1:** 0.82

### Metrics
- Test loss: 0.34  
- Test accuracy: 0.87

### Visual examples
<p float="left">
  <img src="screenshots\training_validation.png" alt="Loss and accuracy curves" width="220" />
  <img src="reports\saliency\sample_0.png" alt="Saliency map overlay" width="220" />
  <img src="reports\occlusion\sample_0.png" alt="Occlusion sensitivity overlay" width="220" />
  <img src="reports\embedding\embedding_tsne.png" alt="t-SNE of embeddings" width="220" />
  <img src="reports\gradcam\sample_0_pred6.png" alt="Grad-CAM overlay" width="220" />
</p>


---

## How to reproduce
1. Install deps:
```bash
pip install -r requirements.txt
```
---
## 🛠 Tech Stack

- **Python 3.9** — Core programming language  
- **TensorFlow 2.19** — Deep learning framework for model building & training  
- **NumPy** — Numerical computations and array handling  
- **Pandas** — Data loading and preprocessing  
- **Matplotlib & Seaborn** — Data visualization and plotting  
- **Jupyter Notebook** — Interactive development environment  
- **Oxford-IIIT Pet Dataset** — Labeled dataset for classification  

---
## 📚 API Reference

### `train_model(dataset_path, epochs=20, batch_size=32)`
- **Description:** Trains the CNN model using the Oxford-IIIT Pet Dataset.  
- **Parameters:**
  - `dataset_path` *(str)* — Path to the dataset folder.
  - `epochs` *(int)* — Number of training epochs (default: 20).
  - `batch_size` *(int)* — Training batch size (default: 32).
- **Returns:** Trained Keras model instance.

### `evaluate_model(model, test_data)`
- **Description:** Evaluates the trained model on test data and prints accuracy/loss.  
- **Parameters:**
  - `model` *(Keras model)* — The trained model object.
  - `test_data` *(tf.data.Dataset)* — Preprocessed test dataset.

### `predict_image(model, image_path)`
- **Description:** Predicts the breed of a given image.  
- **Parameters:**
  - `model` *(Keras model)* — The trained model object.
  - `image_path` *(str)* — Path to the image file.
- **Returns:** Tuple containing:
  - Predicted class label *(str)*
  - Confidence score *(float)*

### CLI Usage
```bash
# Predict a single image
python predict.py --image_path path/to/image.jpg
```
---
## 🤝 Contributing

```bash
# Contributions are welcome! If you’d like to improve this project, follow these steps:

1. Fork the repository

2. Clone your fork locally
git clone https://github.com/your-username/classification-of-pet-faces.git

3. Create a new branch for your feature or fix
git checkout -b feature/your-feature-name

4. Make your changes and commit them
git commit -m "Add: Your meaningful commit message"

5. Push to your branch
git push origin feature/your-feature-name

6. Open a Pull Request and describe your changes

------------------------------------------------
# Contribution Guidelines:
# - Keep commit messages clear and descriptive
# - Follow the existing code style
# - Test your changes before submitting
# - For major changes, open an issue first to discuss
```
---
## Authors

- **Sarthak Aloria** – [SarthakAloria](https://github.com/SarthakAloria)


---
## License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

---
## 🙏 Acknowledgements

- [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/)  
- [TensorFlow Documentation](https://www.tensorflow.org/)  
- [Jupyter Notebook](https://jupyter.org/)  

---
## Support

For any issues or feature requests, please open an issue on GitHub or contact me at sarthakaloria27@gmail.com.

---