from setuptools import setup, find_packages

setup(
    name="llama-architecture-study",
    version="0.1.0",
    description="Modular Llama architecture implementation for text summarization",
    author="Your Name",
    packages=find_packages(),
    install_requires=[
        "tensorflow>=2.10.0",
        "pandas>=1.5.0",
        "numpy>=1.23.0",
        "tokenizers>=0.13.0",
    ],
    python_requires=">=3.8",
)
