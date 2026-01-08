import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

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
    # llm_name = "mistralai/Ministral-3-14B-Instruct-2512"
    # llm_subname = "ministral-3-14b-instruct-2512"
    # llm_name = "google/gemma-3-12b-it"
    # llm_subname = "gemma-3-12b-it"
    # llm_name = "google/gemma-3-27b-it"
    # llm_subname = "gemma-3-27b-it"
    # llm_name = "meta-llama/Llama-3.1-8B-Instruct"
    # llm_subname = "Llama-3.1-8B-Instruct"
    # llm_name = "microsoft/phi-4"
    # llm_subname = "phi-4"
    # llm_name = "Qwen/Qwen3-4B"
    # llm_subname = "Qwen3-4B"
    # llm_name = "Qwen/Qwen3-4B-Instruct-2507"
    # llm_subname = "Qwen3-4B-Instruct-2507"
    # llm_name = "Qwen/Qwen3-30B-A3B-Instruct-2507"
    # llm_subname = "Qwen3-30B-A3B-Instruct-2507"
    # llm_name = "Qwen/Qwen3-32B"
    # llm_subname = "Qwen3-32B"    
    
    # import torch
    # print("GPU count:", torch.cuda.device_count())
    # print("Devices:", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])

    # sampling_params = SamplingParams(
    #     temperature=0, top_p=0.9, max_tokens=3000, stop=["<|eot_id|>"]
    # )

    # llm = LLM(
    #     model=llm_name,
    #     tokenizer_mode="mistral",
    #     config_format="mistral",
    #     load_format="mistral",
    #     # load_format="auto",
    #     # tensor_parallel_size=4,
    #     # dtype="bfloat16", # For Gemma-3
    #     dtype="half",
    #     gpu_memory_utilization=0.95,
    #     max_logprobs=1000,
    #     # device="auto",
    #     max_model_len=25000 # 5000,  # 25000
    # )
    # tokenizer = llm.get_tokenizer()
    
    # llm_name = "mrfakename/mistral-small-3.1-24b-instruct-2503-hf"
    # llm_subname = "Mistral-Small-3.1-24B-Instruct-2503"
    # llm = AutoModelForCausalLM.from_pretrained(
    #     llm_name,
    #     torch_dtype="half",
    #     device_map="auto",
    #     attn_implementation="eager",
    # )
    # tokenizer = AutoTokenizer.from_pretrained(llm_name)

    # llm_name = "./t_Finetuned/tmerged_model_mm_st21pv_k=Just1Valid_full"
    # llm_subname = "Mistral_finetuned_k=Just1Valid_full"

    # llm_name = "./t_Finetuned/tmerged_model_mm_st21pv_k=3_200x2"
    # llm_subname = "Mistral_finetuned_k=3_200x2"
    # llm = AutoModelForCausalLM.from_pretrained(
    #     llm_name, torch_dtype="auto", device_map="auto"
    # )
    # tokenizer = AutoTokenizer.from_pretrained(llm_name)

    ##### Load fine-tuned model #####
    base_model_path = "mrfakename/mistral-small-3.1-24b-instruct-2503-hf"  # base model
    # base_model_path = "google/gemma-3-27b-it"
    adapter_path = "./t_Finetuned/tfinetuned_mm_st21pv_k=3_true_batches=3002x2_lora_v2"  # my adapter folder
    # adapter_path = "./t_Finetuned/tfinetuned_mtsamples_k=3_true_batches=2900x10"
    llm_name = adapter_path

    # llm_subname = "Mistral_finetuned_mtsamples_k=3_true_batches=2900x10"
    llm_subname = "Mistral_finetuned_mm_st21pv_k=3_true_batches=3002x2_lora_v2"
    # Load base model and tokenizer
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype="auto",
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    base_model.resize_token_embeddings(len(tokenizer))
    # Load adapter on top of base model
    adapter = PeftModel.from_pretrained(base_model, adapter_path)
    # Merge adapter into the base model
    llm = adapter.merge_and_unload()

    ######### Model for GPT4o #########
    # llm_name = "gpt-4.1-2025-04-14"
    # llm_subname = "gpt-4.1-2025-04-14"
    # llm_name = "gpt-4.1-mini-2025-04-14"
    # llm_subname = "gpt-4.1-mini-2025-04-14"
    # llm_name = "gpt-4o-2024-11-20"
    # llm_subname = "gpt-4o-2024-11-20"
    # reasoning_effort = None
    # llm_name = "gpt-4.5-preview"
    # llm_subname = "gpt-4.5-preview"
    # llm_name = "o3-mini"
    # llm_subname = "o3-mini"
    # llm_name = "gpt-5-mini"
    # llm_subname = "gpt-5-mini"
    # llm_name = "gpt-5"
    # llm_subname = "gpt-5"
    # reasoning_effort = "medium"

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
        tag2label_ncbi,
        tag2label_bc5cdr,
        tag2label_mtsamples,
        tag2label_vaers,
        extraction_prompts_st21pv_v2,
        extraction_prompts_ncbi,
        extraction_prompts_bc5cdr,
        extraction_prompts_mtsamples,
        extraction_prompts_vaers,
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
    open_ai_api_key = "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"
    # open_ai_api_key = "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"
    openai.api_key = open_ai_api_key

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
    # dataset = "bc5cdr"
    # dataset_dir = "bc5cdr"
    # dataset = "ncbi_disease"
    # dataset_dir = "ncbi_disease"
    # dataset = "mtsamples"
    # dataset_dir = "mtsamples"
    # dataset = "vaers"
    # dataset_dir = "vaers"    

    all_tag2label = {
        "mm_st21pv": tag2label_st21pv,
        "bc5cdr": tag2label_bc5cdr,  # Replace with actual tag2label for bc5cdr
        "ncbi_disease": tag2label_ncbi,  # Replace with actual tag2label for ncbi_disease
        "mtsamples": tag2label_mtsamples,
        "vaers": tag2label_vaers,
    }

    all_extraction_prompts = {
        "mm_st21pv": extraction_prompts_st21pv_v2,
        "bc5cdr": extraction_prompts_bc5cdr,
        "ncbi_disease": extraction_prompts_ncbi,
        "mtsamples": extraction_prompts_mtsamples,
        "vaers": extraction_prompts_vaers,
    }

    extraction_prompts = all_extraction_prompts[dataset]
    tag_to_label = all_tag2label[dataset]

    dynamic = True  # dynamic few-shot (not static prompt)

    version = "v5"

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
    # pmids_test = pmids_test[:40]
    pmids_valid = [pmid for pmid in data if data[pmid]["split"] == "validation"]

    # pmids_test = [
    #     pmid for pmid in data if data[pmid]["split"] == "train"
    # ]  # For predicting "train" split

    pmids_to_eval = {
        "test": pmids_test,
        #  "valid": pmids_valid
    }

    for eval_set in pmids_to_eval:
        pmids_subset = pmids_to_eval[eval_set]
        for k in [1, 3, 5, 10]:
        # for k in [0]:
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
                    prompt_file = "prompts/Prompt_gold_data_no_examples.txt" if k==0 else "prompts/Prompt_gold_data.txt",
                )
                if i == 0 :
                    print('Prompt: ', prompt)

                start_time = time.time()

                ###### GPT ######
                if llm_name in [
                    "gpt-4o-2024-11-20",
                    "gpt-4.5-preview",
                    "o3-mini",
                    "gpt-4.1-mini-2025-04-14",
                    "gpt-4.1-2025-04-14",
                    "gpt-5",
                    "gpt-5-mini",
                    "gpt-5-nano",
                ]:
                    answer = prompt_gpt(
                        system_instructions=system_instructions,
                        gpt_version=llm_name,
                        prompt=prompt,
                        reasoning_effort=reasoning_effort,
                    )

                elif isinstance(llm, LLM):
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
    import multiprocessing
    multiprocessing.set_start_method("spawn", force=True)
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    main()
