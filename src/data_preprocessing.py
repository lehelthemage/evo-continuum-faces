
import os
import zipfile
import gdown
import cv2
import numpy as np
import dlib

# Define dataset paths
CELEBA_URL = 'https://drive.google.com/uc?export=download&id=0B7EVK8r0v71pZjFTYXZWM3FlRnM'
NEANDERTHAL_URL = 'https://drive.google.com/uc?export=download&id=<NEANDERTHAL_FILE_ID>'
FOSSIL_SKULL_URL = '<FOSSIL_SKULL_URL>'

CELEBA_DIR = './data/celeba/'
NEANDERTHAL_DIR = './data/neanderthal_faces/'
FOSSIL_SKULL_DIR = './data/fossil_skulls/'

# Function to download dataset
def download_and_extract(url, zip_file_path, extract_dir):
    if not os.path.exists(extract_dir):
        os.makedirs(extract_dir)

    print(f"Downloading dataset from {url}")
    gdown.download(url, zip_file_path, quiet=False)

    print("Extracting files...")
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    print(f"Dataset extracted to {extract_dir}")

# Function to preprocess images (resize and normalize)
def preprocess_images(input_dir, output_dir, size=(224, 224)):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image_files = [f for f in os.listdir(input_dir) if f.endswith('.jpg')]

    for img_file in image_files:
        img_path = os.path.join(input_dir, img_file)
        img = cv2.imread(img_path)

        # Resize image
        img_resized = cv2.resize(img, size)

        # Normalize the image
        img_normalized = img_resized / 255.0  # Normalize to [0, 1]

        # Save processed image
        output_path = os.path.join(output_dir, img_file)
        cv2.imwrite(output_path, (img_normalized * 255).astype(np.uint8))

    print(f"Images processed and saved to {output_dir}")

# Function to extract facial landmarks from images using dlib
def extract_face_landmarks(image_path):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    landmarks = []

    for face in faces:
        shape = predictor(gray, face)
        landmarks = [(p.x, p.y) for p in shape.parts()]

    return landmarks

# Process CelebA dataset
def preprocess_celeba():
    print("Processing CelebA dataset...")
    zip_file_path = './data/celeba.zip'
    extract_dir = CELEBA_DIR
    download_and_extract(CELEBA_URL, zip_file_path, extract_dir)

    # Process images (resize and normalize)
    processed_images_dir = './data/celeba/processed_images/'
    preprocess_images(extract_dir, processed_images_dir)

    print("CelebA dataset preprocessing complete.")

# Process Neanderthal faces dataset
def preprocess_neanderthal():
    print("Processing Neanderthal faces dataset...")
    zip_file_path = './data/neanderthal_faces.zip'
    extract_dir = NEANDERTHAL_DIR
    download_and_extract(NEANDERTHAL_URL, zip_file_path, extract_dir)

    # Process images (resize and normalize)
    processed_images_dir = './data/neanderthal_faces/processed_images/'
    preprocess_images(extract_dir, processed_images_dir)

    print("Neanderthal faces dataset preprocessing complete.")

# Process Fossil Skulls dataset
def preprocess_fossil_skulls():
    print("Processing Fossil Skulls dataset...")
    zip_file_path = './data/fossil_skulls.zip'
    extract_dir = FOSSIL_SKULL_DIR
    download_and_extract(FOSSIL_SKULL_URL, zip_file_path, extract_dir)

    # Fossil skulls processing could be different depending on the format (e.g., 3D models or 2D images)
    # If they are 3D models, use a 3D processing library such as PyMesh or PyTorch3D to preprocess.

    print("Fossil Skulls dataset preprocessing complete.")

# Main function to preprocess all datasets
def main():
    # Preprocess CelebA
    preprocess_celeba()

    # Preprocess Neanderthal Faces
    preprocess_neanderthal()

    # Preprocess Fossil Skulls
    preprocess_fossil_skulls()

    print("All datasets have been preprocessed.")

if __name__ == "__main__":
    main()
