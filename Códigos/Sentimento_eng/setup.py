from setuptools import setup, find_packages

setup(
    name="sentimento-eng",
    version="1.0.0",
    description="Command-line tool for sentiment analysis of English text paragraphs using BERT and spaCy",
    author="Your Name",
    author_email="your.email@example.com",
    py_modules=["sentimento_eng", "sentimento_detalhado_eng"],
    install_requires=[
        "transformers>=4.21.0",
        "torch>=1.12.0",
        "spacy>=3.4.0",
        "numpy>=1.21.0",
    ],
    entry_points={
        "console_scripts": [
            "sentimento-eng=sentimento_eng:main",
            "sentimento-detalhado-eng=sentimento_detalhado_eng:main",
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
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.7",
)