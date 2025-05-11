import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import multiprocessing

multiprocessing.set_start_method("spawn", force=True)


def main():
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        Gemma3ForConditionalGeneration,
    )
    from vllm import LLM, SamplingParams
    from peft import PeftModel

    # llm_name = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
    # llm_subname = "mistral-small-3.1-24b-instruct-2503-hf"
    llm_name = "google/gemma-3-12b-it"
    llm_subname = "gemma-3-12b-it"
    # llm_name = "google/gemma-3-27b-it"
    # llm_subname = "gemma-3-27b-it"
    # llm_name = "meta-llama/Llama-3.1-8B-Instruct"
    # llm_subname = "Llama-3.1-8B-Instruct"
    # llm_name = "microsoft/phi-4"
    # llm_subname = "phi-4"

    sampling_params = SamplingParams(
        temperature=0, top_p=0.9, max_tokens=3000, stop=["<|eot_id|>"]
    )

    llm = LLM(
        model=llm_name,
        # tokenizer_mode="mistral",
        # config_format="mistral",
        # load_format="mistral",
        load_format="auto",
        # tensor_parallel_size=2,
        # dtype="half",
        # gpu_memory_utilization=0.9,
        max_logprobs=1000,
        device="auto",
        max_model_len=22000,  # 16384
    )
    tokenizer = llm.get_tokenizer()

    # llm_name = "./t_Finetuned/tmerged_model_mm_st21pv_k=Just1Valid_full"
    # llm_subname = "Mistral_finetuned_k=Just1Valid_full"

    # llm_name = "./t_Finetuned/tmerged_model_mm_st21pv_k=3_200x2"
    # llm_subname = "Mistral_finetuned_k=3_200x2"
    # llm = AutoModelForCausalLM.from_pretrained(
    #     llm_name, torch_dtype="auto", device_map="auto"
    # )
    # tokenizer = AutoTokenizer.from_pretrained(llm_name)

    # base_model_path = "mrfakename/mistral-small-3.1-24b-instruct-2503-hf"  # base model
    # adapter_path = "./t_Finetuned/tfinetuned_mm_st21pv_k=1_true_batches=3000x4"  # my adapter folder
    # llm_name = adapter_path
    # llm_subname = "Mistral_finetuned_mm_st21pv_k=1_true_batches=3000x4"
    # # Load base model and tokenizer
    # base_model = AutoModelForCausalLM.from_pretrained(
    #     base_model_path,
    #     torch_dtype="auto",
    #     device_map="auto",
    # )
    # tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    # if tokenizer.pad_token is None:
    #     tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    # base_model.resize_token_embeddings(len(tokenizer))
    # # Load adapter on top of base model
    # adapter = PeftModel.from_pretrained(base_model, adapter_path)
    # # Merge adapter into the base model
    # llm = adapter.merge_and_unload()

    import json
    from thefuzz import fuzz, process
    import re
    import ujson
    import logging
    import inflect
    from collections import Counter, defaultdict
    import numpy as np
    from sentence_transformers import SentenceTransformer, util
    import evaluate
    import spacy
    import ast
    import pandas as pd
    import faiss
    import random

    import torch
    import openai
    import time

    from data_utils import (
        subsets_data,
        tag2label_st21pv,
        extraction_prompts_st21pv_v2,
        format_spans,
        prompt_llm_hf,
        prompt_gpt,
        prompt_vllm,
        topk_examples,
        generate_examples,
        generate_prompt_text,
        extract_last_assistant_message,
        extract_json_from_text,
        parse_answer,
        gather_tagged_entities,
        compute_metrics,
    )

    # seqeval evaluation
    seqeval = evaluate.load("seqeval")
    # spacy tokenizer
    nlp = spacy.blank("en")
    # Create an engine object
    p = inflect.engine()

    # Set up logging configuration
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    seed = 42

    system_instructions = """You are a medical doctor who specializes in clinical trials and observational studies.
    You will act as an expert annotator of research articles provided to you.
    Only answer questions using data explicitly present in given studies.
    """

    annotation_instructions = """Your final output must match the original abstract precisely. Be sure that any errors such as duplicating sentences, omitting text, or introducing hallucinations are corrected before the final output."""

    device_1 = torch.device("cuda:0")
    device_2 = torch.device("cuda:1")

    ######### DATASET #########
    dataset = "mm_st21pv"
    dataset_dir = "mm_st21pv"

    extraction_prompts = extraction_prompts_st21pv_v2
    tag_to_label = tag2label_st21pv

    dynamic = True  # dynamic few-shot (not static prompt)

    version = "v4"

    data_path = f"../data/{dataset_dir}/{dataset}.json"
    data = ujson.load(open(data_path))

    data_train = {k: v for k, v in data.items() if v["split"] == "train"}

    ######### Examples for MM_ST21PV in few-shot #########
    if dynamic:
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

    results = []
    raw_results = []

    pmids_test = [pmid for pmid in data if data[pmid]["split"] == "test"]
    # pmids_test = pmids_test[:542] + pmids_test[543:]
    pmids_valid = [pmid for pmid in data if data[pmid]["split"] == "validation"]
    pmids_valid = pmids_valid[:40]

    # pmids_test = [
    #     pmid for pmid in data if data[pmid]["split"] == "train"
    # ]  # For predicting "train" split

    pmids_to_eval = {
        "test": pmids_test,
        #  "valid": pmids_valid
    }

    for eval_set in pmids_to_eval:
        pmids_subset = pmids_to_eval[eval_set]
        for k in [10]:
            results = []
            raw_results = []

            print(
                # f"Fine-tuned model used : {adapter_path}."
                f"Evaluation on set : {eval_set} with k = {k}"
            )
            # Name of the file to save the results
            file_name = f"{llm_subname}_dynamic={dynamic}_k={k}_{eval_set}Set_{version}"
            nb_failure = 0
            nb_reconstruction_mismatch = 0

            START_TIME = time.time()
            for i, pmid in enumerate(pmids_subset):
                print("i :", i, "pmid :", pmid)
                abstract, true_spans = data[pmid]["input"], data[pmid]["spans"]

                examples, _ = topk_examples(
                    model=model,
                    index=index,
                    query=abstract,
                    corpus=corpus,
                    abstract2pmid=abstract2pmid,
                    data=data,
                    k=k,
                )

                prompt = generate_prompt_text(
                    abstract=abstract,
                    extraction_prompts=extraction_prompts,
                    format_spans=format_spans,
                    topk_examples=examples,
                    annotation_instructions=annotation_instructions,
                    noise=False,
                )

                start_time = time.time()

                if isinstance(llm, LLM):
                    answer = prompt_vllm(
                        system_instructions=system_instructions,
                        llm=llm,
                        tokenizer=tokenizer,
                        sampling_params=sampling_params,
                        prompt=prompt,
                    )
                else:
                    answer = prompt_llm_hf(
                        system_instructions=system_instructions,
                        llm=llm,
                        tokenizer=tokenizer,
                        prompt=prompt,
                    )

                end_time = time.time()
                # Measure and display elapsed time
                elapsed_time = end_time - start_time
                print(f"Elapsed time: {elapsed_time:.2f} seconds")

                raw_result = {"pmid": pmid, "llm_output": answer}
                raw_results.append(raw_result)

                # Save dinamically to check where the error comes from
                with open(
                    f"raw_results/{dataset_dir}/{file_name}.jsonl", "a"
                ) as raw_results_f:
                    raw_results_f.write(json.dumps(raw_result) + "\n")
                print("Answer :", answer)
                final_answer = extract_json_from_text(answer)
                print("final_answer :", final_answer)
                tagged_text = parse_answer(final_answer)

                if tagged_text is None:
                    print(f"Skipping PMID {pmid} due to JSON parsing failure.")
                    nb_failure += 1
                    continue  # Skip this PMID for evaluation

                pred_spans, reconstructed_text, count = gather_tagged_entities(
                    abstract, tagged_text["output"], tag_to_label
                )
                if reconstructed_text != abstract:
                    print(
                        f"Reconstructed text does not match original text for PMID {pmid}."
                    )
                    print("Length original text: ", len(abstract))
                    print("Length reconstructed text: ", len(reconstructed_text))
                    nb_reconstruction_mismatch += 1
                    continue

                final_result = {
                    "pmid": pmid,
                    "tagged_text": tagged_text["output"].strip(),
                    "pred_spans": pred_spans,
                }
                results.append(final_result)

                # if i >= 0 :
                #     break

            # Final save
            with open(f"results/{dataset_dir}/{file_name}.json", "w") as f:
                json.dump(results, f)
            with open(f"raw_results/{dataset_dir}/{file_name}.json", "w") as f:
                json.dump(raw_results, f)

            llm_output = results
            pmids = [el["pmid"] for el in llm_output]
            pmid2pred_spans = {
                entry["pmid"]: entry["pred_spans"] for entry in llm_output
            }

            results_metrics, df, nb_valid_eval, empty = compute_metrics(
                pmids=pmids,
                data=data,
                tag_to_label=tag_to_label,
                pmid2pred_spans=pmid2pred_spans,
                debug=True,
            )

            print("Number of pmids that we tried to evaluate: ", len(pmids_test))
            print("Number of valid evaluations: ", nb_valid_eval)
            print("Number of empty pred_spans: ", empty)
            print("Results : ", results_metrics)

            END_TIME = time.time()
            print(
                f"Total time: {END_TIME - START_TIME:.2f} seconds for processing {len(pmids_test)} abstracts"
            )

            print("Path to data that were used in the few-shot prompt :", data_path)
            print(f"Results saved in:", f"results/{dataset_dir}/{file_name}.json")
            print("Annotations Instructions in prompt: ", annotation_instructions)


if __name__ == "__main__":
    main()
