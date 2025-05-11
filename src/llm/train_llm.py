import pickle
import ujson
import json
import sys
import os
from collections import Counter, defaultdict
from typing import List, Dict, Union
from itertools import islice
from thefuzz import fuzz, process
import re
import logging
import inflect
import evaluate
import spacy
import ast
import faiss
import time
import pandas as pd
import numpy as np
import torch
import random
import gzip
from tqdm import tqdm
from typing import Optional
from sentence_transformers import SentenceTransformer, util
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizer,
    AutoConfig,
)
from peft import LoraConfig, get_peft_model, PeftModel
from data_utils import *

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
import matplotlib.pyplot as plt

# Check GPU availability
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(torch.cuda.current_device())
    print(f"GPU is available: {gpu_name}")
    device = torch.device("cuda")
else:
    print("No GPU is available. Using CPU instead.")
    device = torch.device("cpu")

max_length = {
    "mm_st21pv": 9000,
}


class DatasetNER(Dataset):
    def __init__(
        self,
        data: List[Dict],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 9000,
        split: str = "train",
    ):
        self.data = [data[pmid] for pmid in data if data[pmid]["split"] == split]
        self.tokenizer = tokenizer

        self.max_length = max_length

        if isinstance(data, torch.utils.data.Subset):
            self.indices = data.indices
        else:
            self.indices = list(range(len(data)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        system_instructions = """You are a medical doctor who specializes in clinical trials and observational studies.
        You will act as an expert annotator of research articles provided to you.
        Only answer questions using data explicitly present in given studies."""

        item = self.data[idx]
        prompt = item["prompt"]
        answer = item["output"]
        wrapped_answer = (
            "```json\n" + json.dumps({"output": answer}, indent=2) + "\n```"
        )
        # Apply chat template for consistency with inference
        messages = [
            {"role": "system", "content": system_instructions},
            {"role": "user", "content": prompt},
            {
                "role": "assistant",
                "content": wrapped_answer,
            },
        ]
        chat_input = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

        # Tokenize the input
        encoding = self.tokenizer(
            chat_input,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        # Mask out the prompt (excluding assistant's content) in the labels
        chat_input = self.tokenizer.apply_chat_template(messages[:-1], tokenize=False)
        tokenized_input = self.tokenizer(chat_input, return_tensors="pt")
        prompt_length = tokenized_input["input_ids"].size(1)
        # print("prompt_length : ", prompt_length)

        label = input_ids.clone()
        label[:prompt_length] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": label,
        }


def main():
    device_1 = torch.device("cuda:0")

    ######### DATASET #########
    # dataset_name = "mm_st21pv"
    # dataset_dir = "mm_st21pv"
    # extraction_prompts = extraction_prompts_st21pv_v2
    # tag_to_label = tag2label_st21pv

    # dataset_name = "ncbi_disease"
    # dataset_dir = "ncbi_disease"
    # extraction_prompts = extraction_prompts_ncbi
    # tag_to_label = tag2label_ncbi

    dataset_name = "bc5cdr"
    dataset_dir = "bc5cdr"
    extraction_prompts = extraction_prompts_bc5cdr
    tag_to_label = tag2label_bc5cdr

    data_path = f"../data/{dataset_name}/{dataset_name}.json"
    data = ujson.load(open(data_path))
    pmids_test = [pmid for pmid in data if data[pmid]["split"] == "test"]
    pmids_test = pmids_test[:40]
    pmids_valid = [pmid for pmid in data if data[pmid]["split"] == "validation"]
    pmids_valid = pmids_valid[:40]  # (40 abstracts takes 30mins)
    system_instructions = """You are a medical doctor who specializes in clinical trials and observational studies.
    You will act as an expert annotator of research articles provided to you.
    Only answer questions using data explicitly present in given studies.
    """
    annotation_instructions = """Your final output must match the original abstract precisely. Be sure that any errors such as duplicating sentences, omitting text, or introducing hallucinations are corrected before the final output."""

    dynamic = True  # dynamic few-shot (not static prompt)
    version = "v1"

    data_train = {k: v for k, v in data.items() if v["split"] == "train"}

    ######### Examples for MM_ST21PV in few-shot #########
    corpus = [data_train[pmid]["input"] for pmid in data_train]
    pmid2abstract = {pmid: data_train[pmid]["input"] for pmid in data_train}
    abstract2pmid = {data_train[pmid]["input"]: pmid for pmid in data_train}

    model = SentenceTransformer("princeton-nlp/sup-simcse-bert-base-uncased")
    model.to(device_1)
    # Generate embeddings for the corpus
    corpus_embeddings = model.encode(corpus, convert_to_tensor=True)
    corpus_embeddings = corpus_embeddings.cpu().detach().numpy()
    embedding_dimension = corpus_embeddings.shape[1]

    # Create the HNSW index with the correct arguments
    M = 32  # Number of neighbors in the HNSW graph
    index = faiss.IndexHNSWFlat(embedding_dimension, M)
    # Normalize the corpus embeddings if using cosine similarity
    faiss.normalize_L2(corpus_embeddings)
    # Add the embeddings to the index
    index.add(corpus_embeddings)
    # Print the number of sentences added to the index
    print(f"Number of sentences in the index: {index.ntotal}")

    best_f1_score = 0.0

    llm_name = "mrfakename/mistral-small-3.1-24b-instruct-2503-hf"

    # Load base model (use float16, not float8)
    llm = AutoModelForCausalLM.from_pretrained(
        llm_name, torch_dtype=torch.float16, device_map="auto"
    )

    # Enable gradient checkpointing BEFORE LoRA
    llm.gradient_checkpointing_enable()

    # Load tokenizer and add pad token if needed
    tokenizer = AutoTokenizer.from_pretrained(llm_name)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    llm.resize_token_embeddings(len(tokenizer))

    # LoRA configuration
    lora_config = LoraConfig(
        r=128,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        target_modules=["q_proj", "v_proj"],
    )

    llm = get_peft_model(llm, lora_config)
    llm.print_trainable_parameters()

    # base_model_path = "mrfakename/mistral-small-3.1-24b-instruct-2503-hf"  # base model
    # adapter_path = "./t_Finetuned/tfinetuned_mm_st21pv_k=1_true_batches=3000x4"  # my adapter folder
    # # Load base model and tokenizer
    # base_model = AutoModelForCausalLM.from_pretrained(
    #     base_model_path,
    #     torch_dtype="auto",
    #     device_map="auto",
    # )

    # # Enable gradient checkpointing BEFORE LoRA
    # base_model.gradient_checkpointing_enable()

    # tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    # if tokenizer.pad_token is None:
    #     tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    # base_model.resize_token_embeddings(len(tokenizer))
    # # Load adapter on top of base model
    # adapter = PeftModel.from_pretrained(base_model, adapter_path)
    # # Merge adapter into the base model
    # llm = adapter.merge_and_unload()

    # Load dataset
    k = 3
    with open(f"../data_finetune_llm/{dataset_name}_k={k}_true.json") as f:
        dataset = json.load(f)

    train_dataset = DatasetNER(
        data=dataset, split="train", tokenizer=tokenizer, max_length=5000
    )
    # valid_dataset = DatasetNER(
    #     data=dataset, split="validation", tokenizer=tokenizer, max_length=2500
    # )

    batch_size = 3
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    print("Number of training batches : ", len(train_dataloader))

    optimizer = torch.optim.Adam(llm.parameters(), lr=1e-4)

    nb_epochs = 15
    batches_limit = 3000
    llm_subname = (
        f"tfinetuned_{dataset_name}_k={k}_true_batches={batches_limit}x{batch_size}"
    )
    save_dir = f"./t_Finetuned/{llm_subname}"
    save_dir2 = f"./t_Finetuned/last_epoch_{llm_subname}"

    for epoch in range(nb_epochs):
        start_time = time.time()
        llm.train()
        for idx, batch in enumerate(train_dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = llm(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )

            loss = outputs.loss
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            print(f"Epoch: {epoch+1}, Batch:{idx}, Training Loss: {loss.item()}")
            del loss, outputs  # save some space
            torch.cuda.empty_cache()

            if idx == batches_limit:
                break

        # Time logging
        elapsed_time = time.time() - start_time
        h, m = divmod(int(elapsed_time), 3600)
        m, s = divmod(m, 60)
        print(f"Time for one epoch: {h}h {m}m {s}s")

        # After each epoch, evaluate on one batch of validation data
        llm.eval()
        results = []
        if epoch >= 5:
            with torch.no_grad():
                nb_failure = 0
                nb_reconstruction_mismatch = 0
                for i, pmid in enumerate(pmids_valid):
                    abstract, true_spans = data[pmid]["input"], data[pmid]["spans"]
                    true_spans = fuse_adjacent_tag(true_spans)

                    examples, _ = topk_examples(
                        model=model,
                        index=index,
                        query=abstract,
                        corpus=corpus,
                        abstract2pmid=abstract2pmid,
                        data=data,
                        k=k,
                        most_similar=True,
                    )

                    prompt = generate_prompt_text(
                        abstract=abstract,
                        extraction_prompts=extraction_prompts,
                        format_spans=format_spans,
                        topk_examples=examples,
                        annotation_instructions=annotation_instructions,
                        noise=False,
                    )

                    if i == 0:
                        print("The prompt is : ", prompt)

                    prompt1 = dataset[pmid]["prompt"]
                    if prompt != prompt1:
                        print(
                            f"Prompt mismatch for PMID {pmid} "
                            f"Generated: {prompt} "
                            f"Expected: {prompt1}"
                        )

                    answer = prompt_llm_hf(
                        system_instructions=system_instructions,
                        llm=llm,
                        tokenizer=tokenizer,
                        prompt=prompt,
                    )
                    print("answer : ", answer)
                    answer2 = extract_last_assistant_message(answer)
                    final_answer = extract_last_json_from_text(answer2)
                    print("final_answer :", final_answer)
                    tagged_text = parse_answer(final_answer)

                    if tagged_text is None:
                        print(f"PMID {pmid} has JSON parsing failure.")
                        nb_failure += 1
                        tagged_text = {}
                        tagged_text["output"] = ""

                    pred_spans, reconstructed_text, count = gather_tagged_entities(
                        abstract, tagged_text["output"], tag_to_label
                    )
                    pred_spans = [
                        p
                        for p in pred_spans
                        if (p["start"] != -1 and (p["end"] - p["start"] < 70))
                    ]
                    pred_spans = sorted(pred_spans, key=lambda x: x["start"])
                    pred_spans = fuse_adjacent_tag(pred_spans)

                    if reconstructed_text != abstract:
                        print(
                            f"Reconstructed text does not match original text for PMID {pmid}."
                        )
                        print("Length riginal text: ", len(abstract))
                        print("Length reconstructed text: ", len(reconstructed_text))
                        nb_reconstruction_mismatch += 1

                    final_result = {
                        "pmid": pmid,
                        "tagged_text": tagged_text["output"].strip(),
                        "pred_spans": pred_spans,
                    }
                    results.append(final_result)

                llm_output = results
                pmids = [el["pmid"] for el in llm_output]
                pmid2pred_spans = {
                    entry["pmid"]: entry["pred_spans"] for entry in llm_output
                }

                results, df, nb_valid_eval, empty = compute_metrics(
                    pmids=pmids,
                    data=data,
                    tag_to_label=tag_to_label,
                    pmid2pred_spans=pmid2pred_spans,
                    debug=True,
                )
                # Compute the average validation loss across the 10 batches
                f1_score = results["f1"]

                print("Number of pmids that we tried to evaluate: ", len(pmids_valid))
                print("Number of valid evaluations: ", nb_valid_eval)
                print("Number of empty pred_spans: ", empty)
                print(
                    f"Number of failure : {nb_failure}| Number of reconstruction mismatch : {nb_reconstruction_mismatch}"
                )
                print("Results on validation set: ", results)

                print(f"Epoch: {epoch+1}, Average Validation F1 score : {f1_score}")

                # If this is the best validation f1 score so far, save the model
                if f1_score > best_f1_score:
                    best_f1_score = f1_score
                    llm.save_pretrained(save_dir)
                    tokenizer.save_pretrained(save_dir)
                    llm.config.save_pretrained(
                        save_dir
                    )  # Need to save the config as well for loading the model in vllm
                    print(
                        f"New best model saved with Validation F1 score: {best_f1_score}"
                    )

                    # if best_f1_score >= 0.82:
                    #     for k2 in [0, 1, 3]:
                    #         results = []
                    #         raw_results = []

                    #         print(f"Evaluation with k = {k2}")
                    #         # Name of the file to save the results
                    #         file_name = (
                    #             f"{llm_subname}_dynamic={dynamic}_k={k2}_{version}"
                    #         )
                    #         nb_failure = 0
                    #         nb_reconstruction_mismatch = 0
                    #         START_TIME = time.time()
                    #         for i, pmid in enumerate(pmids_test):
                    #             abstract, true_spans = (
                    #                 data[pmid]["input"],
                    #                 data[pmid]["spans"],
                    #             )
                    #             true_spans = fuse_adjacent_tag(true_spans)

                    #             examples, _ = topk_examples(
                    #                 model=model,
                    #                 index=index,
                    #                 query=abstract,
                    #                 corpus=corpus,
                    #                 abstract2pmid=abstract2pmid,
                    #                 data=data,
                    #                 k=k2,
                    #                 most_similar=True,
                    #             )

                    #             prompt = generate_prompt_text(
                    #                 abstract=abstract,
                    #                 extraction_prompts=extraction_prompts,
                    #                 format_spans=format_spans,
                    #                 topk_examples=examples,
                    #                 annotation_instructions=annotation_instructions,
                    #                 noise=False,
                    #             )

                    #             start_time = time.time()

                    #             ##### HUGGING FACE ######
                    #             answer = prompt_llm_hf(
                    #                 system_instructions=system_instructions,
                    #                 llm=llm,
                    #                 tokenizer=tokenizer,
                    #                 prompt=prompt,
                    #             )
                    #             end_time = time.time()

                    #             # Measure and display elapsed time
                    #             elapsed_time = end_time - start_time
                    #             print(f"Elapsed time: {elapsed_time:.2f} seconds")

                    #             print("i :", i, "pmid :", pmid)
                    #             raw_result = {"pmid": pmid, "llm_output": answer}
                    #             raw_results.append(raw_result)

                    #             # Save dinamically to check where the error comes from
                    #             with open(
                    #                 f"raw_results/{dataset_dir}/{file_name}.jsonl", "a"
                    #             ) as raw_results_f:
                    #                 raw_results_f.write(json.dumps(raw_result) + "\n")

                    #             print("Answer :", answer)
                    #             final_answer = extract_json_from_text(answer)
                    #             print("final_answer :", final_answer)
                    #             tagged_text = parse_answer(final_answer)

                    #             if tagged_text is None:
                    #                 print(
                    #                     f"Skipping PMID {pmid} due to JSON parsing failure."
                    #                 )
                    #                 nb_failure += 1
                    #                 # continue  # Skip this PMID for evaluation

                    #             pred_spans, reconstructed_text, count = (
                    #                 gather_tagged_entities(
                    #                     abstract, tagged_text["output"], tag_to_label
                    #                 )
                    #             )
                    #             pred_spans = [
                    #                 p
                    #                 for p in pred_spans
                    #                 if (
                    #                     p["start"] != -1
                    #                     and (p["end"] - p["start"] < 70)
                    #                 )
                    #             ]
                    #             pred_spans = sorted(
                    #                 pred_spans, key=lambda x: x["start"]
                    #             )
                    #             pred_spans = fuse_adjacent_tag(pred_spans)
                    #             if reconstructed_text != abstract:
                    #                 print(
                    #                     f"Reconstructed text does not match original text for PMID {pmid}."
                    #                 )
                    #                 print("Length original text: ", len(abstract))
                    #                 print(
                    #                     "Length reconstructed text: ",
                    #                     len(reconstructed_text),
                    #                 )
                    #                 nb_reconstruction_mismatch += 1
                    #                 # continue

                    #             final_result = {
                    #                 "pmid": pmid,
                    #                 "tagged_text": tagged_text["output"].strip(),
                    #                 "pred_spans": pred_spans,
                    #             }
                    #             results.append(final_result)

                    #         END_TIME = time.time()
                    #         print(
                    #             f"Total time: {END_TIME - START_TIME:.2f} seconds for processing {len(pmids_test)} abstracts"
                    #         )

                    #         # Final save
                    #         with open(
                    #             f"results/{dataset_dir}/{file_name}.json", "w"
                    #         ) as f:
                    #             json.dump(results, f)
                    #         with open(
                    #             f"raw_results/{dataset_dir}/{file_name}.json", "w"
                    #         ) as f:
                    #             json.dump(raw_results, f)

                    #         llm_output = results
                    #         pmids = [el["pmid"] for el in llm_output]
                    #         pmid2pred_spans = {
                    #             entry["pmid"]: entry["pred_spans"]
                    #             for entry in llm_output
                    #         }

                    #         results, df, nb_valid_eval, empty = compute_metrics(
                    #             pmids=pmids,
                    #             data=data,
                    #             tag_to_label=tag_to_label,
                    #             pmid2pred_spans=pmid2pred_spans,
                    #             debug=True,
                    #         )

                    #         print(
                    #             "Number of pmids that we tried to evaluate: ",
                    #             len(pmids_test),
                    #         )
                    #         print("Number of valid evaluations: ", nb_valid_eval)
                    #         print("Number of empty pred_spans: ", empty)
                    #         print(
                    #             f"Number of failure : {nb_failure}| Number of reconstruction mismatch : {nb_reconstruction_mismatch}"
                    #         )
                    #         print("Results on test set: ", results)

                    #         print(
                    #             "Path to data that were used in the few-shot prompt :",
                    #             data_path,
                    #         )
                    #         print(
                    #             f"Results saved in:",
                    #             f"results/{dataset_dir}/{file_name}.json",
                    #         )
                    #         print(
                    #             "Annotations Instructions in prompt: ",
                    #             annotation_instructions,
                    #         )

    print(f"Training complete. Best Validation F1 score: {best_f1_score}")

    # Save the fine-tuned model of the last epoch
    llm.save_pretrained(save_dir2)
    tokenizer.save_pretrained(save_dir2)
    llm.config.save_pretrained(save_dir2)


if __name__ == "__main__":
    main()
