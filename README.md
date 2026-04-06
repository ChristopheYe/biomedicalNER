# Where do LLMs currently stand on biomedical NER in both clean and noisy settings ?

## Citation
If you have found our manuscript useful in your work, please consider citing:
> @inproceedings{ye-mitchell-2026-llms,
    title = "Where do {LLM}s currently stand on biomedical {NER} in both clean and noisy settings ?",
    author = "Ye, Christophe  and
      Mitchell, Cassie S.",
    editor = "Demberg, Vera  and
      Inui, Kentaro  and
      Marquez, Llu{\'i}s",
    booktitle = "Findings of the {A}ssociation for {C}omputational {L}inguistics: {EACL} 2026",
    month = mar,
    year = "2026",
    address = "Rabat, Morocco",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.findings-eacl.51/",
    doi = "10.18653/v1/2026.findings-eacl.51",
    pages = "977--1001",
    ISBN = "979-8-89176-386-9"
}

## Installation

In order to run this, please install the following conda environment : 

```bash
conda create -n bioner python=3.10
conda activate bioner
pip install -e .
```

## Data
The `data/` directory contains all datasets formatted for training and inference with LLMs.

The `data2/` directory contains all datasets formatted for training and inference with BERT-based models.

## Models

### LLMs
The `llm/` directory contains scripts for:
Training: `train_llm.py`
Inference: `inference3.py`

### BERT-based models
`models` directory contains the scripts for training (`train_perso.py`)
