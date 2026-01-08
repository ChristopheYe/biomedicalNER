from setuptools import find_packages, setup
import os

here = os.path.abspath(os.path.dirname(__file__))


setup(
    name="biomedical_NER",
    description="LLM for biomedical named entity recognition",
    # long_description=long_description,
    url="",
    author="Anonymous",
    author_email="",
    keywords=[
        "biomedical",
        "noisy dataset",
        "named entity recognition",
    ],
    packages=find_packages(),
    python_requires=">= 3.9",
    install_requires=[
        "tqdm",
        "pandas",
        "numpy>=1.23,<2",
        "matplotlib",
        "ujson",
        "torch",
        "transformers",
        "faiss-gpu",
        "sentence_transformers",
        "vllm",
        "openai",
        "wandb",
        "inflect",
        "evaluate",
        "datasets",
        "thefuzz",
        "spacy",
        "peft",
        "seqeval",
        "bioc",
    ],
)
