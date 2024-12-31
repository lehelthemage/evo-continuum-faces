import os
import requests
from zipfile import ZipFile
from io import BytesIO
from pathlib import Path
from PIL import Image
import numpy as np
import random
from tqdm import tqdm
import gdown


# Helper function to download files
def download_file(url, save_path):
    """Download a file from a URL and save it to the given path."""
    print(f"Downloading from {url}...")
    response = requests.get(url)
    response.raise_for_status()  # Will raise an error if the download failed
    with open(save_path, 'wb') as file:
        file.write(response.content)
    print(f"Downloaded to {save_path}")


# Function to extract CelebA dataset
def download_and_extract_celebA():
    """Downloads the CelebA dataset and extracts it using gdown."""
    celeba_url = "https://drive.google.com/uc?export=download&id=0B7EVK8r0v71pZjFTYXZsZ2ZpZ2s"
    output_dir = Path("data/celeba")
    output_dir.mkdir(parents=True, exist_ok=True)
    zip_file_path = output_dir / "celeba.zip"

    # Download using gdown (handles Google Drive download more gracefully)
    gdown.download(celeba_url, str(zip_file_path), quiet=False)

    # Extract the dataset
    with ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(output_dir)

    print("CelebA dataset extracted successfully!")


# Function to download other datasets (e.g., animal images)
def download_and_extract_animal_dataset():
    """Placeholder function to download and extract an animal dataset."""
    # Define a URL for the animal dataset (replace with actual URL)
    animal_url = "https://example.com/animal-dataset.zip"
    output_dir = Path("data/animals")
    output_dir.mkdir(parents=True, exist_ok=True)
    zip_file_path = output_dir / "animal_dataset.zip"

    # Download and unzip the animal dataset
    download_file(animal_url, zip_file_path)
    with ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(output_dir)

    print("Animal dataset extracted successfully!")


# Function to preprocess images (resize and normalize)
def preprocess_images(dataset_path, image_size=(128, 128)):
    """Preprocess images by resizing and normalizing them."""
    image_paths = list(Path(dataset_path).glob("**/*.jpg"))
    preprocessed_dir = Path("data/preprocessed")
    preprocessed_dir.mkdir(parents=True, exist_ok=True)

    print(f"Preprocessing {len(image_paths)} images...")

    for image_path in tqdm(image_paths):
        try:
            # Open and resize the image
            with Image.open(image_path) as img:
                img = img.resize(image_size)
                img = np.array(img) / 255.0  # Normalize the image
                # Save the preprocessed image to the new folder
                save_path = preprocessed_dir / image_path.name
                Image.fromarray((img * 255).astype(np.uint8)).save(save_path)

        except Exception as e:
            print(f"Error processing {image_path}: {e}")

    print(f"Preprocessing complete. Images saved to {preprocessed_dir}")


# Function to split dataset into training and testing sets
def split_dataset(dataset_path, train_size=0.8):
    """Splits the dataset into training and testing sets."""
    image_paths = list(Path(dataset_path).glob("**/*.jpg"))
    random.shuffle(image_paths)
    
    split_index = int(len(image_paths) * train_size)
    
    train_paths = image_paths[:split_index]
    test_paths = image_paths[split_index:]
    
    # Create directories for train and test
    train_dir = Path("data/train")
    test_dir = Path("data/test")
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    # Move the images to the respective directories
    for path in tqdm(train_paths):
        os.rename(path, train_dir / path.name)

    for path in tqdm(test_paths):
        os.rename(path, test_dir / path.name)

    print(f"Dataset split into {len(train_paths)} training and {len(test_paths)} testing images.")


if __name__ == "__main__":
    # Step 1: Download datasets (e.g., CelebA, animal dataset)
    download_and_extract_celebA()
    download_and_extract_animal_dataset()

    # Step 2: Preprocess images
    preprocess_images("data/celeba")
    preprocess_images("data/animals")

    # Step 3: Split the dataset into training and testing
    split_dataset("data/preprocessed")
