from setuptools import setup, find_packages

setup(
    name='evo-continuum-faces',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy', 
        'pandas', 
        'torch', 
        'torchvision', 
        'flask', 
        'dlib', 
        'opencv-python', 
        'scikit-learn', 
        'matplotlib', 
        'jupyter',
        'requests',
        'tqdm',
        'pillow',
        'tensorflow',
        'keras'
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="A project to simulate human face evolution using GANs",
    url="https://github.com/lehelthemage/evolution-of-human-faces",
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',
)