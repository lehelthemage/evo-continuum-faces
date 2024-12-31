# Evolutionary Face Generator

## Overview
The Evolutionary Face Generator project uses **StyleGAN2** to simulate an evolutionary continuum of human faces, transitioning through evolutionary stages starting from modern humans and going back through primates, hominins, and early mammals. The project will morph faces gradually, simulating the process of evolution through generations, with each generation representing a "deep fake" of a different animal or ancestor.

This project provides a way to visualize evolutionary changes in facial features over time by using deep learning models trained on various datasets, including datasets of modern humans, hominins, and other primates. It uses a "dial" system where users can slide through generations, observing how human faces would evolve into earlier ancestors.

## Features
- Generate evolutionary morphs between modern human faces and early mammalian ancestors.
- Utilize a transition model built on StyleGAN2 to create deep fakes of ancestral faces.
- Preprocess datasets from various sources, including human faces, hominins, and animal face datasets.
- Explore the evolutionary continuum through interactive generation at different speeds.

## Requirements
To run the project, you'll need the following dependencies:

- **Python 3.x**
- **TensorFlow 2.x** (for training the model)
- **PyTorch 1.12.1** (for running StyleGAN2)
- **NumPy**
- **OpenCV**
- **Pandas**
- **Matplotlib**
- **TQDM** (progress bar)

To install all dependencies, run:

```bash
pip install -r requirements.txt
```

## Setup

Follow these steps to get started with the project.
1. Clone the Repository

To get started, clone the repository to your local machine:

git clone https://github.com/yourusername/Evolutionary-Face-Generator.git
cd Evolutionary-Face-Generator

2. Install the Required Dependencies

Install all the required Python libraries from requirements.txt:

pip install -r requirements.txt

3. Prepare the Datasets

You will need datasets of human, hominin, and animal faces. Below is a list of datasets to get started with:

    CelebA (for modern human faces)
    FFHQ (high-quality human face images)
    Hominin and Primate faces (collected manually from various sources)

Download and place the datasets in the data/ folder in the following structure:

data/
    human_faces/
        celebA/
        ffhq/
    hominin_faces/
        (Homo erectus, Homo habilis, Neanderthal, etc.)
    animal_faces/
        (primate, mammal, etc.)

4. Data Preprocessing

Before training, you need to preprocess the datasets.
Resize and Normalize Images

The images should be resized to a resolution of 1024x1024 and normalized to the range of [-1, 1]. This can be done using the scripts/data_preprocessing.py script:

python scripts/data_preprocessing.py

This will process and save images in the data/processed folder. If there are any specific modifications needed for the data (e.g., removing images with artifacts or aligning faces), add those steps to this script.
Convert Images to TFRecords Format

StyleGAN2 requires datasets to be in TFRecords format. Use the scripts/dataset_tool.py script to convert your images into the correct format:

python scripts/dataset_tool.py --source=data/processed/human_faces --dest=data/tfrecords/human_faces.tfrecords
python scripts/dataset_tool.py --source=data/processed/hominin_faces --dest=data/tfrecords/hominin_faces.tfrecords
python scripts/dataset_tool.py --source=data/processed/animal_faces --dest=data/tfrecords/animal_faces.tfrecords

This will convert the images into TFRecord files, which can be used for training the model.
5. Train the StyleGAN2 Model

After the data is prepared and converted to TFRecords, you can begin training the StyleGAN2 model. The training script is located in scripts/run_training.py.

Run the following command to start the training process:

python scripts/run_training.py --data_dir=data/tfrecords --config=config-f --gpus=1 --mirror_augment=true --batch_size=8

Where:

    --data_dir: Path to the directory containing the TFRecord files.
    --config: Configuration file for StyleGAN2. Use config-f for a balanced configuration.
    --gpus: The number of GPUs to use for training. Set it to the number of GPUs available (e.g., --gpus=1).
    --mirror_augment: Augmentation option to mirror the images during training.
    --batch_size: The batch size for training.

During training, checkpoints will be saved in the models/ directory. You can resume training from a checkpoint if interrupted by specifying the checkpoint file as --network.
6. Generate Images from the Trained Model

Once training is complete, you can generate images using the trained model. Use the scripts/run_generator.py script to generate images from the trained model:

python scripts/run_generator.py generate-images --network=models/trained_model.pkl --num_images=10

Where:

    --network: Path to the trained model checkpoint (e.g., models/trained_model.pkl).
    --num_images: The number of images to generate. You can adjust this value based on your needs.

This will generate images based on the trained model, allowing you to visualize the evolutionary transitions.
7. Visualize and Interact with the Evolutionary Faces

For future work, we plan to implement an interactive "dial" system where you can drag a slider and observe the faces transition from modern human to an early mammalian ancestor. In the meantime, you can explore and visualize the generated faces manually.
Notebooks

We also provide Jupyter notebooks for data collection, preprocessing, and visualizing the generated faces.

    notebooks/data_collection.ipynb: A notebook to guide you through collecting and preprocessing datasets. It includes steps for downloading and resizing the images.
    notebooks/generate_faces.ipynb: A notebook to visualize the face generation process from the trained model. You can experiment with adjusting parameters and generating faces at different evolutionary stages.

Contributing

If you would like to contribute to this project, feel free to fork the repository, make your changes, and submit a pull request. Contributions are welcome to enhance the data collection process, improve the model, or build the user interface.
License

This project is licensed under the MIT License - see the LICENSE file for details.
Acknowledgements

    The StyleGAN2 architecture was developed by NVIDIA.
    Special thanks to the contributors and dataset providers for making this project possible.

