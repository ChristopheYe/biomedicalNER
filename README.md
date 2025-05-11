# Where do LLMs currently stand on biomedical NER in both clean and noisy settings ?

## Installation

In order to run this, please install the following conda environment : 

```bash
conda create -n bioner python=3.9
conda activate bioner
pip install -e .
```

## Data
`data` repository contains all datasets in the format for LLM training/inference
`data2` repository contains all datasets in the format for BERT-based models training/inference

## Models

### LLMs
`llm` repository contains the scripts for training (`train_llm`) and inference (`inference3.py`)

### BERT-based models
`models` repository contains the scripts for training (`train_perso`)
