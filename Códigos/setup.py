from setuptools import setup, find_packages

setup(
    name="sentimento",
    version="1.0.0",
    description="Command-line tool for sentiment analysis of text paragraphs in Portuguese using BERT",
    author="Your Name",
    author_email="your.email@example.com",
    py_modules=["sentimento", "sentimento_detalhado"],  # Adicionar aqui
    install_requires=[
        "transformers>=4.21.0",
        "torch>=1.12.0",
        "spacy>=3.4.0",
        "numpy>=1.21.0",
    ],
    entry_points={
        "console_scripts": [
            "sentimento=sentimento:main",
            "sentimento_detalhado=sentimento_detalhado:main",  # Adicionar aqui
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.7",
)
