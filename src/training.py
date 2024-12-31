
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from .model import build_generator, build_discriminator
from .data_preprocessing import load_images_from_directory, preprocess_data

def train_gan(images, epochs=10000, batch_size=64, latent_dim=100, image_shape=(128, 128, 3)):
    x_train, _, _, _ = preprocess_data(images, None)
    generator = build_generator(latent_dim)
    discriminator = build_discriminator(image_shape)
    
    discriminator.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5), metrics=['accuracy'])
    
    discriminator.trainable = False
    z = layers.Input(shape=(latent_dim,))
    img = generator(z)
    validity = discriminator(img)
    combined = tf.keras.Model(z, validity)
    combined.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))
    
    for epoch in range(epochs):
        idx = np.random.randint(0, x_train.shape[0], batch_size)
        real_images = x_train[idx]
        labels_real = np.ones((batch_size, 1))
        
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        fake_images = generator.predict(noise)
        labels_fake = np.zeros((batch_size, 1))
        
        d_loss_real = discriminator.train_on_batch(real_images, labels_real)
        d_loss_fake = discriminator.train_on_batch(fake_images, labels_fake)
        
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        g_loss = combined.train_on_batch(noise, labels_real)
        
        if epoch % 1000 == 0:
            print(f"{epoch} [D loss: {0.5 * np.add(d_loss_real, d_loss_fake)}] [G loss: {g_loss}]")
