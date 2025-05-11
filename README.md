# Where do LLMs currently stand on biomedical NER in both clean and noisy settings ?

## Installation

In order to run this, please install the following conda environment : 

```bash
conda create -n bioner python=3.9
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
`models` directory contains the scripts for training (`train_perso`)
