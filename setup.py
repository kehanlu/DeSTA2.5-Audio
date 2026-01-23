from setuptools import setup, find_packages

setup(
    name='desta',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        "whisper_normalizer",
        "huggingface_hub",
        "lulutils @ https://github.com/kehanlu/lulutils/archive/refs/heads/main.zip",
        "transformers>=4.49.0",
        "safetensors",
        "librosa",
        "peft",
        "faster-whisper"
    ],
    description='DeSTA2.5-Audio',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)