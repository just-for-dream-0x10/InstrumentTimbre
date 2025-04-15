from setuptools import setup, find_packages

setup(
    name="instrument_timbre_model",
    version="0.1.0",
    author="TimWood0x10",
    author_email="busanlang@gmail.com",
    description="A musical instrument timbre conversion model based on deep learning",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "torch>=1.9.0",
        "torchaudio>=0.9.0",
        "librosa>=0.8.1",
        "soundfile>=0.10.3",
        "pedalboard>=0.5.8",
    ],
    extras_require={
        "dev": [
            "matplotlib>=3.4.0",
            "tqdm>=4.62.0",
            "scikit-learn>=1.0.0",
            "pytest>=6.2.5",
        ],
    },
)
