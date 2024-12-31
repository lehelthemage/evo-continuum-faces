
import os
import matplotlib.pyplot as plt

def save_image(image, path):
    plt.imsave(path, image)

def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
