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

    ######### Model for Qwen/Llama/Phi/Mistral #########

    llm_name = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
    llm_subname = "Mistral-Small-3.1-24B-Instruct-2503"

    sampling_params = SamplingParams(
        temperature=0, top_p=0.9, max_tokens=4000, stop=["<|eot_id|>"]
    )
    llm = LLM(
        model=llm_name,
        tokenizer_mode="mistral",
        config_format="mistral",
        load_format="mistral",
        tensor_parallel_size=2,
        dtype="half",
        # gpu_memory_utilization=0.6,
        max_logprobs=1000,
        device="auto",
        max_model_len=18000,
    )

    tokenizer = llm.get_tokenizer()

    # llm_name = "mrfakename/mistral-small-3.1-24b-instruct-2503-hf"
    # llm_subname = "Mistral-Small-3.1-24B-Instruct-2503"
    # llm = AutoModelForCausalLM.from_pretrained(
    #     llm_name,
    #     torch_dtype="half",
    #     device_map="auto",
    #     # attn_implementation="eager",
    # )
    # tokenizer = AutoTokenizer.from_pretrained(llm_name)

    # llm_name = "meta-llama/Llama-3.1-8B-Instruct"
    # llm_subname = "Llama-3.1-8B-Instruct"
    # llm_name = "mistralai/Mistral-Nemo-Instruct-2407"
    # llm_subname = "Mistral-Nemo-Instruct-2407"
    # llm_name = "microsoft/phi-4"
    # llm_subname = "phi-4"
    # llm_name = "google/gemma-3-12b-it"
    # llm_subname = "gemma-3-12b-it"

    # llm_name = "Qwen/Qwen2.5-72B-Instruct"
    # llm_subname = "Qwen2.5-72B-Instruct"

    # llm_name = "Qwen/QwQ-32B"
    # llm_subname = "QwQ-32B"

    # llm_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
    # llm_subname = "DeepSeek-R1-Distill-Qwen-14B"

    ######### Model for Qwen/QwQ #########
    # llm_name = "Qwen/QwQ-32B-Preview"
    # file_name = "QwQ-32B-Preview"
    # reasoning = True

    # llm_name = "google/gemma-3-12b-it"
    # llm_subname = "gemma-3-12b-it"
    # llm = Gemma3ForConditionalGeneration.from_pretrained(
    #     llm_name,
    #     attn_implementation="eager",
    #     torch_dtype="auto",
    #     device_map=device_1,
    # )
    # llm = AutoModelForCausalLM.from_pretrained(
    #     llm_name, torch_dtype="auto", device_map="auto"
    # )
    # tokenizer = AutoTokenizer.from_pretrained(llm_name)

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
    import argparse
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

    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Run inference with different parameters."
    )

    # Arguments from CLI
    parser.add_argument("--k", type=int, required=True, help="Number of k")
    parser.add_argument("--noise_level", type=float, required=True, help="Noise level")
    parser.add_argument("--subset", type=int, default=None, help="Subset")

    # Parse arguments
    args = parser.parse_args()

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

    # annotation_instructions = """As shown in the examples, **many texts require tagging**—**do not hesitate** to mark every instance where a mention fits the categories. **If a text matches any category, it must be tagged**. Do not omit valid mentions.
    # Your task is to **modify the original abstract** by adding "@@mentions##entity@@" **wherever required**.
    # Avoid errors such as duplicating sentences, omitting text, or introducing hallucinations. Ensure the output is precise, consistent, and faithfully preserves the original structure.
    # """

    annotation_instructions = """Your final output must match the original abstract precisely. Be sure that any errors such as duplicating sentences, omitting text, or introducing hallucinations are corrected before the final output.
    """

    # annotation_instructions = """A student has already started the annotations so that it goes faster for you. However, it's the student work so he probably introduced a lot of mistakes. Your task is to correct all his mistakes.
    # """

    # annotation_instructions = """A student has already begun the annotations to speed up the process for you. However, since it's student work, the annotations contain a significant amount of noise (approximately 50%). Your task is to carefully review and correct all errors.  REVISIONS ARE MANDATORY, as the student has made numerous errors.
    # """

    device_1 = torch.device("cuda:0")
    device_2 = torch.device("cuda:1")

    ######### DATASET #########
    # dataset = "bc5cdr"
    # extraction_prompts = extraction_prompts_bc5cdr
    # tag_to_label = tag_to_label_bc5cdr

    # dataset = "mm_st21pv"
    # dataset_dir = "mm_st21pv"
    # dataset_v2 = "mm_st21pv_BioLinkBERT-base_test_v1"

    dist = "N"
    noise_level = args.noise_level
    # dataset = f"mm_st21pv_BioLinkBERT-base_train_v1_noise={noise_level}_dist={dist}_v1"
    # dataset_dir = f"mm_st21pv_BioLinkBERT_noise_{int(noise_level*100)}pc"

    dataset = f"mm_st21pv_phi-4_v16_noise={noise_level}_dist={dist}_v1"
    dataset_dir = f"mm_st21pv_phi-4_noise_{int(noise_level*100)}pc"

    # dataset = f"mm_st21pv_gemma-3-12b-it_v2_noise={noise_level}_dist={dist}_v1"
    # dataset_dir = f"mm_st21pv_gemma-3-12b-it_noise_{int(noise_level*100)}pc"

    # dataset = f"mm_st21pv_gpt-4o-2024-11-20_dynamic=True_k=10_v11_noise={noise_level}_dist={dist}_v1"
    # dataset_dir = f"mm_st21pv_gpt4o_noise_{int(noise_level*100)}pc"

    # dataset = f"mm_st21pv_Mistral-Small-3.1-24B-Instruct-2503_dynamic=True_k=5_v3_noise={noise_level}_dist={dist}_v1"
    # dataset_dir = (
    #     f"mm_st21pv_Mistral-Small-3.1-24B-Instruct-2503_noise_{int(noise_level*100)}pc"
    # )

    # noise = True if "noise_level" in globals() else 0

    extraction_prompts = extraction_prompts_st21pv_v2
    tag_to_label = tag2label_st21pv

    dynamic = True  # dynamic few-shot (not static prompt)
    reasoning = True
    k = args.k
    most_similar = True
    subset = args.subset
    print("subset : ", subset)

    version = "v1"

    # Name of the file to save the results
    if subset is not None:
        file_name = (
            f"{llm_subname}_dynamic={dynamic}_k={k}_{version}_clean_subset={subset}"
        )
    else:
        file_name = f"{llm_subname}_dynamic={dynamic}_k={k}_{version}"

    data_path = f"/home2/cye73/noisyNER/noisyNER/data/mm_st21pv/{dataset}.json"
    data = ujson.load(open(data_path))

    if subset is not None:
        print(f"Only use {subset} abstracts")
        data_train = {k: v for k, v in data.items() if k in subsets_data[subset]}
    else:
        print("Use all abstracts")
        data_train = {k: v for k, v in data.items() if v["split"] == "train"}
        # data_train = {
        #     k: v for k, v in data.items() if v["split"] in {"valid", "test"}
        # }  # For predicting "train" split

    if "dataset_v2" in globals():
        data_path_v2 = f"/home2/cye73/noisyNER/noisyNER/data/{dataset_v2}.json"
        data_v2 = ujson.load(open(data_path_v2))

    ######### Examples for MM_ST21PV in few-shot #########
    if dynamic:
        corpus = [data_train[pmid]["input"] for pmid in data_train]
        pmid2abstract = {pmid: data_train[pmid]["input"] for pmid in data_train}
        abstract2pmid = {data_train[pmid]["input"]: pmid for pmid in data_train}

        model = SentenceTransformer("princeton-nlp/sup-simcse-bert-base-uncased")
        # model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
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
    pmids_test = pmids_test[:40]
    # pmids_test = pmids_test[:2] + pmids_test[3:40]

    # pmids_test = [pmid for pmid in data if data[pmid]["split"] == "train"] # For predicting "train" split

    START_TIME = time.time()
    for i, pmid in enumerate(pmids_test):
        if "dataset_v2" not in globals():
            abstract, true_spans = data[pmid]["input"], data[pmid]["spans"]
        else:
            abstract, abstract2, true_spans = (
                data[pmid]["input"],
                data_v2[pmid]["output"],
                data[pmid]["spans"],
            )  # For already annotated from BERT based-model

        if dynamic:
            examples, _ = topk_examples(
                model=model,
                index=index,
                query=abstract,
                corpus=corpus,
                abstract2pmid=abstract2pmid,
                data=data,
                k=k,
                most_similar=most_similar,
            )
        else:
            examples = generate_examples(data=data_train, k=k, seed=seed)

        prompt = generate_prompt_text(
            abstract=abstract,
            extraction_prompts=extraction_prompts,
            format_spans=format_spans,
            topk_examples=examples,
            annotation_instructions=annotation_instructions,
            noise=False,
        )

        start_time = time.time()

        ###### GPT ######
        if llm_name in ["gpt-4o-2024-11-20", "gpt-4.5-preview", "o3-mini"]:
            answer = prompt_gpt(
                system_instructions=system_instructions,
                gpt_version=llm_name,
                prompt=prompt,
                reasoning_effort="medium",
            )

        ###### VLLM ######
        elif isinstance(llm, LLM):
            answer = prompt_vllm(
                system_instructions=system_instructions,
                llm=llm,
                tokenizer=tokenizer,
                sampling_params=sampling_params,
                prompt=prompt,
            )
        ###### HF ######
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

        print("i :", i, "pmid :", pmid)
        raw_result = {"pmid": pmid, "llm_output": answer}
        raw_results.append(raw_result)

        # Save dinamically to check where the error comes from
        with open(f"raw_results/{dataset_dir}/{file_name}.jsonl", "a") as raw_results_f:
            raw_results_f.write(json.dumps(raw_result) + "\n")

        # Process the answer
        final_answer = extract_json_from_text(answer)
        print("final_answer : ", final_answer)
        if not final_answer:
            final_answer = answer
        tagged_text = parse_answer(final_answer)

        if tagged_text is None:
            print(f"Skipping PMID {pmid} due to JSON parsing failure.")
            continue  # Skip this iteration if parsing fails

        pred_spans, reconstructed_text, count = gather_tagged_entities(
            abstract, tagged_text["output"], tag_to_label
        )
        if reconstructed_text != abstract:
            print(f"Reconstructed text does not match original text for PMID {pmid}.")
            print("Length riginal text: ", len(abstract))
            print("Length reconstructed text: ", len(reconstructed_text))
            continue

        final_result = {
            "pmid": pmid,
            "tagged_text": tagged_text["output"].strip(),
            "pred_spans": pred_spans,
        }
        results.append(final_result)

    END_TIME = time.time()
    print(
        f"Total time: {END_TIME - START_TIME:.2f} seconds for processing {len(pmids_test)} abstracts"
    )

    # Final save
    with open(f"results/{dataset_dir}/{file_name}.json", "w") as f:
        json.dump(results, f)
    with open(f"raw_results/{dataset_dir}/{file_name}.json", "w") as f:
        json.dump(raw_results, f)

    llm_output = results
    pmids = [el["pmid"] for el in llm_output]
    pmid2pred_spans = {entry["pmid"]: entry["pred_spans"] for entry in llm_output}

    results, df, nb_valid_eval, empty = compute_metrics(
        pmids=pmids,
        data=data,
        tag_to_label=tag_to_label,
        pmid2pred_spans=pmid2pred_spans,
        debug=True,
    )

    print("Number of pmids that we tried to evaluate: ", len(pmids_test))
    print("Number of valid evaluations: ", nb_valid_eval)
    print("Number of empty pred_spans: ", empty)
    print("Results : ", results)

    print("Path to data that were used in the few-shot prompt :", data_path)
    print(f"Results saved in:", f"results/{dataset_dir}/{file_name}.json")
    print("Annotations Instructions in prompt: ", annotation_instructions)


if __name__ == "__main__":
    main()
