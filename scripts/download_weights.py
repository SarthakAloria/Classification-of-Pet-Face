import os
import urllib.request

MODEL_URL = "https://drive.google.com/file/d/1VnAS62ISF3F_79ce5qzIan5R2yMElBYU/view?usp=drive_link"
MODEL_PATH = os.path.join("saved_models", "best_model.h5")

def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading model...")
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Download complete!")
    else:
        print("Model already exists.")

if __name__ == "__main__":
    download_model()
