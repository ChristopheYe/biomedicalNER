"""
Train and evaluate NER models on MM-ST21PV / bc5cdr dataset
"""

import numpy as np
import evaluate
import warnings
import ujson
import os
import sys
import torch

# Ignore all warnings
warnings.filterwarnings("ignore")

from data_module_perso import DatasetNER, create_label2id
from data_utils import *
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from transformers import DataCollatorForTokenClassification
from torch.utils.data import DataLoader, ConcatDataset, Subset
from functools import partial
from transformers.integrations import WandbCallback

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load package to evaluate models
seqeval = evaluate.load("seqeval")


def compute_metrics(p, id2label, mode="eval"):
    """
    Compute precision, recall, and f1 for each class and overall
    """
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [id2label[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    # print(results)

    class_specific_f1 = {
        k: v["f1"] for k, v in results.items() if not k.startswith("overall")
    }

    if mode == "eval":
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
            "class_specific_f1": class_specific_f1,
            "detailed_results": results,
        }
    else:
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
            "class_specific_results": results,
        }


# devices
print("Is CUDA available?", torch.cuda.is_available())
print("Device count:", torch.cuda.device_count())
print("Current device:", torch.cuda.current_device())
print(
    "Device name:",
    (
        torch.cuda.get_device_name(torch.cuda.current_device())
        if torch.cuda.is_available()
        else "No GPU detected"
    ),
)

hf_model = "michiyasunaga/BioLinkBERT-base"
model_subname = "BioLinkBERT-base"
# hf_model = "answerdotai/ModernBERT-base"
# model_subname = "ModernBERT-base"
# hf_model = "answerdotai/ModernBERT-large"
# model_subname = "ModernBERT-large"
# hf_model = "FacebookAI/roberta-large"
# model_subname = "roberta-large"
# hf_model = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
# model_subname = "PubMedBERT-base"
# hf_model = "dmis-lab/biobert-base-cased-v1.2"
# model_subname = "BioBERT-base"

# subset = 100  # Using only a subset of the data
# dataset_name = "mm_st21pv"
# dataset_name = "ebm_pico"
# dataset_name = "pico_extraction"
# dataset_name = "ncbi_disease"
dataset_name = "bc5cdr"
version = "v2"

# # For loading already trained models
# llm = "mistral3"
# noise = 0.1
# key = f"{llm}_noise={noise}"
# model_subname = model_params[key]["model_subname"]
# checkpoint = model_params[key]["checkpoint"]
# local_path = f"outputs/{dataset_name}_{model_subname}/{checkpoint}"
# print("local_path:", local_path)


## For training the model with noisy dataset
# noise = 1
# distribution = "N"
# dataset_name = f"mm_st21pv_Mistral-Small-3.1-24B-Instruct-2503_dynamic=True_k=5_v3_noise={noise}_dist={distribution}_v1"
# dataset_name = f"mm_st21pv_gpt-4o-2024-11-20_dynamic=True_k=10_v11_noise={noise}_dist={distribution}_v1"
# dataset_name = f"mm_st21pv_phi-4_v16_noise={noise}_dist={distribution}_v1"
# dataset_name = f"mm_st21pv_gemma-3-12b-it_v2_noise={noise}_dist={distribution}_v1"

if "subset" in globals():
    output_dir = (
        f"outputs/{dataset_name}_{model_subname}_clean_subset={subset}_{version}"
    )
else:
    output_dir = f"outputs/{dataset_name}_{model_subname}_{version}"


# dataset_name = "ontonotes"

max_length = 512  # 1024
batch_size = 16

tokenizer = AutoTokenizer.from_pretrained(hf_model)

data_path = f"/home2/cye73/noisyNER/noisyNER/data2/{dataset_name}/{dataset_name}.json"
data = ujson.load(open(data_path))

tag2label_full = {
    "mm_st21pv": tag2label_st21pv,
    "bc5cdr": tag2label_bc5cdr,
    "ebm_pico": tag2label_pico,
    "pico_extraction": tag2label_pico,
    "ncbi_disease": tag2label_ncbi,
}
tags = [tag for tag in tag2label_full[dataset_name]]

if "subset" in globals():
    # data_train = [item for item in data if item["pmid"] in subsets_data[subset]]
    data_train = [item for item in data if item["split"] == "train"][:subset]
else:
    data_train = [item for item in data if item["split"] == "train"]

# Load model
if "local_path" in globals() and os.path.exists(local_path):
    model = AutoModelForTokenClassification.from_pretrained(
        local_path, trust_remote_code=True
    )
    print("Loaded model from local path:", local_path)
    label2id = model.config.label2id
    id2label = {v: k for k, v in label2id.items()}
    num_labels = len(id2label)
else:
    label2id = create_label2id(data, tags)
    id2label = {v: k for k, v in label2id.items()}
    num_labels = len(id2label)
    print("label2id :", label2id)
    model = AutoModelForTokenClassification.from_pretrained(
        hf_model, num_labels=len(id2label), id2label=id2label, label2id=label2id
    )
    print("Loaded model from HuggingFace:", hf_model)

# Create datasets for each split
train_dataset = DatasetNER(
    data_train, tokenizer, tags, label2id, split="train", max_length=max_length
)
valid_dataset = DatasetNER(
    data, tokenizer, tags, label2id, split="validation", max_length=max_length
)
test_dataset = DatasetNER(
    data, tokenizer, tags, label2id, split="test", max_length=max_length
)

collate_fn = DataCollatorForTokenClassification(tokenizer, padding=True)

print("Model num_labels:", model.config.num_labels)
print("Model config label2id:", model.config.label2id)

print("label2id size:", len(label2id))
print("label2id :", label2id)

model.config.num_labels = len(id2label)

# Train model
training_args = TrainingArguments(
    output_dir=output_dir,
    learning_rate=5e-4,  # 1e-4,  # 5e-4, # 2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=30,  # 50,  # 30 # 100
    weight_decay=0.01,
    warmup_ratio=0.2,
    evaluation_strategy="epoch",
    save_strategy="epoch",  # Save checkpoints at the end of each epoch
    save_total_limit=1,  # Keep only the best checkpoint
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    logging_strategy="epoch",
    report_to="wandb",
)

# combined_dataset = ConcatDataset([valid_dataset, test_dataset])  # Merges datasets

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    # train_dataset=combined_dataset,
    # eval_dataset=train_dataset,
    tokenizer=tokenizer,
    data_collator=collate_fn,
    compute_metrics=partial(compute_metrics, id2label=id2label),
)

trainer.train()

# Predict output and save metrics
# output = trainer.predict(train_dataset)
output = trainer.predict(test_dataset)

labels = (
    output.label_ids.tolist() if output.label_ids is not None else None
)  # ground truth labels
model_labels = np.argmax(output.predictions, axis=-1).tolist()  # model labels
metrics = output.metrics

print("Metrics for the whole test set:", metrics)

# Construct save path
model_name = hf_model.split("/")[-1]
saving_path = f"results/dataset={dataset_name}_model={model_name}.json"
os.makedirs(os.path.dirname(saving_path), exist_ok=True)

# Save everything properly
results = {
    "metrics": metrics,
    "model_labels": model_labels,
    "labels": labels,  # Include ground truth labels if available
}

with open(saving_path, "w") as f:
    ujson.dump(results, f, indent=2)

print(
    f"Results for dataset {dataset_name} using {model_name} were saved in : {saving_path}"
)


test_dataset_40 = Subset(test_dataset, range(40))

output_40 = trainer.predict(test_dataset_40)

labels_40 = (
    output_40.label_ids.tolist() if output_40.label_ids is not None else None
)  # ground truth labels
model_labels_40 = np.argmax(output_40.predictions, axis=-1).tolist()  # model labels
metrics_40 = output_40.metrics

print("Metrics for the first 40 abstracts in test set:", metrics_40)
