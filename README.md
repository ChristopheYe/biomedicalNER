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
    ISBN = "979-8-89176-386-9",
    abstract = "Biomedical Named Entity Recognition (NER) consists of identifying and classifying important biomedical entities mentioned in text. Traditionally, biomedical NER has heavily relied on domain-specific pre-trained language models; particularly variant of BERT models. With the emergence of large language models (LLMs), some studies have evaluated their performance on biomedical NLP tasks. These studies consistently show that, despite their general capabilities, LLMs still fall short compared to specialized BERT-based models for biomedical NER. However, as LLMs continue to advance at a remarkable pace, natural questions arise: Are they still far behind, or are they starting to be competitive? In this study, we investigate the performance of recent LLMs across multiple biomedical NER datasets under both clean and noisy dataset conditions. Our findings reveal that LLMs are progressively closing the performance gap with BERT-based models and demonstrate particular strengths in low-data settings. Moreover, our results suggest that in-context learning with LLMs exhibits a notable degree of robustness to noise, making them a promising alternative in settings where labeled data is scarce or noisy."
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
