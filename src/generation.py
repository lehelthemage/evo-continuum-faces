
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from .model import build_generator

def generate_face(latent_dim=100, model_path="generator_model.h5"):
    generator = tf.keras.models.load_model(model_path)
    noise = np.random.normal(0, 1, (1, latent_dim))
    generated_image = generator.predict(noise)
    plt.imshow(generated_image[0])
    plt.axis('off')
    plt.show()
