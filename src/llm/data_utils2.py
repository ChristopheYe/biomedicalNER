import openai
import json
from thefuzz import fuzz, process
import re
import ujson
import logging
import inflect
from collections import Counter, defaultdict
import numpy as np
import evaluate
import spacy
import ast
import pandas as pd
from difflib import SequenceMatcher
import faiss
import random
from itertools import chain

from vllm.inputs.data import TokensPrompt

# seqeval evaluation
seqeval = evaluate.load("seqeval")

# spacy tokenizer
nlp = spacy.blank("en")

# Create an engine object
p = inflect.engine()

all_distributions = {
    "wl": {
        "wrong_label": 0.5,
        "wrong_overlap": 0.125,
        "multiple_entities": 0.125,
        "invalid_tag": 0.125,
        "missing": 0.125,
    },
    "m": {
        "missing": 0.5,
        "wrong_overlap": 0.125,
        "multiple_entities": 0.125,
        "invalid_tag": 0.125,
        "wrong_label": 0.125,
    },
    "wo": {
        "wrong_overlap": 0.5,
        "wrong_label": 0.125,
        "multiple_entities": 0.125,
        "invalid_tag": 0.125,
        "missing": 0.125,
    },
    "me": {
        "multiple_entities": 0.5,
        "wrong_label": 0.125,
        "wrong_overlap": 0.125,
        "invalid_tag": 0.125,
        "missing": 0.125,
    },
    "it": {
        "invalid_tag": 0.5,
        "wrong_label": 0.125,
        "wrong_overlap": 0.125,
        "multiple_entities": 0.125,
        "missing": 0.125,
    },
}

tag2label_bc5cdr = {"Chemical": 0}

extraction_prompts_bc5cdr = {
    "Chemical": "Return the text that is related to chemical. Example usage: in 'Suxamethonium infusion rate and observed fasciculations.', return 'Suxamethonium'."
}

tag2label_ncbi = {
    "Disease": 0,
}

extraction_prompts_ncbi = {
    "Disease": "Return the text that is related to disease. Example usage: in 'Suxamethonium infusion rate and observed fasciculations.', return 'fasciculations'.",
}

typeName2ID_st21pv = {
    "health_care_activity": "T058",  #
    "research_activity": "T062",  #
    "injury_or_poisoning": "T037",  #
    "biologic_function": "T038",  #
    "virus": "T005",  #
    "bacterium": "T007",  #
    "eukaryote": "T204",  #
    "anatomical_structure": "T017",  #
    "medical_device": "T074",  #
    "body_substance": "T031",  #
    "chemical": "T103",  #
    "food": "T168",  #
    "clinical_attribute": "T201",  #
    "finding": "T033",  #
    "spatial_concept": "T082",  #
    "body_system": "T022",  #
    "biomedical_occupation_or_discipline": "T091",  #
    "organization": "T092",  #
    "professional_or_occupational_group": "T097",  #
    "population_group": "T098",  #
    "intellectual_product": "T170",  #
}

tag2label_st21pv = {
    "T058": 0,
    "T062": 1,
    "T037": 2,
    "T038": 3,
    "T005": 4,
    "T007": 5,
    "T204": 6,
    "T017": 7,
    "T074": 8,
    "T031": 9,
    "T103": 10,
    "T168": 11,
    "T201": 12,
    "T033": 13,
    "T082": 14,
    "T022": 15,
    "T091": 16,
    "T092": 17,
    "T097": 18,
    "T098": 19,
    "T170": 20,
}

extraction_prompts_st21pv_v2 = {
    "T058": "Healthcare Activity : Return the name of the health care activity described. Example usage: in 'a pilot study of an evidence-based psychological intervention', return 'intervention'.",
    "T062": "Research Activity : Return the name of the research activity mentioned. Example usage: in 'By using exome sequencing and extreme phenotype design', return 'exome sequencing'.",
    "T037": "Injury or Poisoning : Return the name of the injury or poisoning mentioned. Example usage: in 'and their toxic effects on aquatic species have been reported', return 'toxic effects'.",
    "T038": "Biologic Function : Return the name of the biologic function described. Example usage: in 'DCTN4 as a modifier of chronic Pseudomonas aeruginosa infection in cystic fibrosis', return 'chronic Pseudomonas aeruginosa infection' and 'cystic fibrosis'.",
    "T005": "Virus : Return the name of the virus mentioned. Example usage: in 'Flaviviruses, including Zika and dengue (DENV), pose a serious global threat to human health.', return 'Flaviviruses', 'Zika', 'dengue' and 'DENV'.",
    "T007": "Bacterium : Return the name of the bacterium mentioned. Example usage: in 'for the identification of lactic acid bacteria', return 'lactic acid bacteria'.",
    "T204": "Eukaryote : Return the name of the eukaryotic organism mentioned. Example usage: in 'as an inert carrier was investigated against Sitophilus oryzae', return 'Sitophilus oryzae'.",
    "T017": "Anatomical Structure : Return the name of the anatomical structure mentioned. Example usage: in 'Polymerase chain reaction and direct sequencing were used to screen DNA samples for DCTN4 variants.', return 'DNA samples'.",
    "T074": "Medical_device : Return the name of the medical device mentioned. Example usage: in 'important future components for bionanoelectronic devices', return 'bionanoelectronic devices'.",
    "T031": "Body_substance : Return the name of the body substance mentioned. Example usage: in 'Apoptosis has been shown to be induced by serum deprivation or copper treatment.', return 'serum'.",
    "T103": "Chemical : Return the name of the chemical mentioned. Example usage: in 'DCTN4 as a modifier of chronic Pseudomonas aeruginosa infection in cystic fibrosis', return 'DCTN4'.",
    "T168": "Food : Return the name of the food mentioned. Example usage: in 'with no detrimental effects on grain quality', return 'grain'.",
    "T201": "Clinical Attribute : Return the name of the clinical attribute described. Example usage: in 'Anthropometric measurements, including height, weight, waist circumference', return 'waist circumference'.",
    "T033": "Finding : Return the name of the finding mentioned. Example usage: in 'chronic Pa infection (CPA) is associated with reduced lung function, faster rate of lung decline', return 'faster rate of lung decline'.",
    "T082": "Spatial Concept : Return the name of the spatial concept mentioned. Example usage: in 'The difference in structure of the two compounds', return 'structure'.",
    "T022": "Body system : Return the name of the body system mentioned. Example usage: in 'reduce the potential risk of skin ulceration', return 'skin'.",
    "T091": "Biomedical Occupation or Discipline : Return the name of the biomedical occupation or discipline mentioned. Example usage: in 'its symptoms are broad and place patients at crossroads between dermatology, hematology', return 'dermatology' and 'hematology'.",
    "T092": "Organization : Return the name of the organization mentioned. Example usage: in 'in a cohort of adult CF patients from a single centre', return 'centre'.",
    "T097": "Professional or Occupational Group : Return the name of the professional or occupational group mentioned. Example usage: in 'Delivered by a health psychologist', return 'health psychologist'.",
    "T098": "Population Group : Return the name of the population group mentioned. Example usage: in 'in a cohort of adult CF patients from a single centre', return 'cohort'.",
    "T170": "Intellectual Product : Return the name of the intellectual product mentioned. Example usage: in 'We designed our intervention using effective components of behaviour change interventions informed by psychological theory', return 'psychological theory'.",
}

extraction_prompts_st21pv_gpt = {
    "health_care_activity": "Definition of 'Health Care Activity' : An activity of or relating to the practice of medicine or involving the care of patients. Return the name of the health care activity described. Example usage: in 'the patient underwent a surgical procedure', return 'surgical procedure'.",
    "research_activity": "Definition of 'Research Activity' : An activity carried out as part of research or experimentation. Return the name of the research activity mentioned. Example usage: in 'the research involved a randomized controlled trial', return 'randomized controlled trial'.",
    "injury_or_poisoning": "Definition of 'Injury or Poisoning' : A traumatic wound, injury, or poisoning caused by an external agent or force. Return the name of the injury or poisoning mentioned. Example usage: in 'the patient suffered from a traumatic brain injury', return 'traumatic brain injury'.",
    "biologic_function": "Definition of 'Biologic Function' : A state, activity or process of the body or one of its systems or parts. Return the name of the biologic function described. Example usage: in 'the regulation of cell growth was studied', return 'regulation of cell growth'.",
    "virus": "Definition of 'Virus' : An organism consisting of a core of a single nucleic acid enclosed in a protective coat of protein. A virus may replicate only inside a host living cell. A virus exhibits some but not all of the usual characteristics of living things. Return the name of the virus mentioned. Example usage: in 'the infection was caused by the influenza virus', return 'influenza virus'.",
    "bacterium": "Definition of 'Bacterium' : A small, typically one-celled, prokaryotic micro-organism. Return the name of the bacterium mentioned. Example usage: in 'the infection was caused by Escherichia coli', return 'Escherichia coli'.",
    "eukaryote": "Definition of 'Eukaryote' : One of the three domains of life (the others being Bacteria and Archaea), also called Eukarya. These are organisms whose cells are enclosed in membranes and possess a nucleus. They comprise almost all multicellular and many unicellular organisms, and are traditionally divided into groups (sometimes called kingdoms) including Animals, Plants, Fungi, various Algae, and other taxa that were previously part of the old kingdom Protista. Return the name of the eukaryotic organism mentioned. Example usage: in 'the study focused on Saccharomyces cerevisiae', return 'Saccharomyces cerevisiae'.",
    "anatomical_structure": "Definition of 'Anatomical Structure' : A normal or pathological part of the anatomy or structural organization of an organism. Return the name of the anatomical structure mentioned. Example usage: in 'the tumor was located in the liver', return 'liver'.",
    "medical_device": "Definition of 'Medical Device' : A manufactured object used primarily in the diagnosis, treatment, or prevention of physiologic or anatomic disorders. Return the name of the medical device mentioned. Example usage: in 'the patient was fitted with a pacemaker', return 'pacemaker'.",
    "body_substance": "Definition of 'Body Substance' : Extracellular material, or mixtures of cells and extracellular material, produced, excreted, or accreted by the body. Included here are substances such as saliva, dental enamel, sweat, and gastric acid. Return the name of the body substance mentioned. Example usage: in 'elevated levels of hemoglobin were noted', return 'hemoglobin'.",
    "chemical": "Definition of 'Chemical' : Compounds or substances of definite molecular composition. Chemicals are viewed from two distinct perspectives in the network, functionally and structurally. Almost every chemical concept is assigned at least two types, generally one from the structure hierarchy and at least one from the function hierarchy. Return the name of the chemical mentioned. Example usage: in 'the solution contained sodium chloride', return 'sodium chloride'.",
    "food": "Definition of 'Food' : Any substance generally containing nutrients, such as carbohydrates, proteins, and fats, that can be ingested by a living organism and metabolized into energy and body tissue. Some foods are naturally occurring, others are either partially or entirely made by humans. Return the name of the food mentioned. Example usage: in 'the diet included soybeans', return 'soybeans'.",
    "clinical_attribute": "Definition of 'Clinical Attribute' : An observable or measurable property or state of an organism of clinical interest. Return the name of the clinical attribute described. Example usage: in 'the patient exhibited elevated blood pressure', return 'elevated blood pressure'.",
    "finding": "Definition of 'Finding' : That which is discovered by direct observation or measurement of an organism attribute or condition, including the clinical history of the patient. The history of the presence of a disease is a 'Finding' and is distinguished from the disease itself. Return the name of the finding mentioned. Example usage: in 'chronic Pa infection (CPA) is associated with reduced lung function, faster rate of lung decline', return 'faster rate of lung decline'.",
    "spatial_concept": "Definition of 'Spatial Concept' : A location, region, or space, generally having definite boundaries. Return the name of the spatial concept mentioned. Example usage: in 'the lesion was located in the left hemisphere', return 'left hemisphere'.",
    "body_system": "Definition of 'Body System' : A complex of anatomical structures that performs a common function. Return the name of the body system mentioned. Example usage: in 'reduce the potential risk of skin ulceration', return 'skin'.",
    "biomedical_occupation_or_discipline": "Definition of 'Biomedical Occupation or Discipline' : A vocation, academic discipline, or field of study related to biomedicine. Return the name of the biomedical occupation or discipline mentioned. Example usage: in 'the field of oncology has seen rapid advancements', return 'oncology'.",
    "organization": "Definition of 'Organization' : The result of uniting for a common purpose or function. The continued existence of an organization is not dependent on any of its members, its location, or particular facility. Components or subparts of organizations are also included here. Although the names of organizations are sometimes used to refer to the buildings in which they reside, they are not inherently physical in nature. Return the name of the organization mentioned. Example usage: in 'the study was conducted by the World Health Organization', return 'World Health Organization'.",
    "professional_or_occupational_group": "Definition of 'Professional or Occupational Group' : An individual or individuals classified according to their vocation. Return the name of the professional or occupational group mentioned. Example usage: in 'the study surveyed pediatricians', return 'pediatricians'.",
    "population_group": "Definition of 'Population Group' : An indivdual or individuals classified according to their sex, racial origin, religion, common place of living, financial or social status, or some other cultural or behavioral attribute. Return the name of the population group mentioned. Example usage: in 'the study focused on adolescents', return 'adolescents'.",
    "intellectual_product": "Definition of 'Intellectual Product' : A conceptual entity resulting from human endeavor. Concepts assigned to this type generally refer to information created by humans for some purpose. Return the name of the intellectual product mentioned. Example usage: in 'the findings were published in the New England Journal of Medicine', return 'New England Journal of Medicine'.",
}

format_spans = "@@text##entity@@"


def retrieve_abstract_and_spans(data_path, pmid):
    """
    Function to retrieve the abstract and spans from the .json file
    ----------
    path_to_processed_for_modeling : str
        The path to the dataset.json file
    pmid : int
        The pmid of the article to retrieve the spans
    """
    with open(data_path, "r") as file:
        datasets = ujson.load(file)
    spans = None
    for article in datasets:
        if article["pmid"] == pmid:
            abstract = article["text"]
            spans = article["spans"]
            break
    if spans is None:
        logging.info("pmid not found in the list of documents")
    return abstract, spans


def knn_query(model, index, query, k=3, most_similar=True):
    """
    Find the top k most similar embeddings of the query from the corpus.
    ------
    model : SentenceTransformer model
    index : faiss index
    query : str (mention + surrounding context)
    k : int (number of similar embeddings to find)
    most_similar : bool (if True, return the most similar, else return the least similar)
    """
    # Generate embedding for the query
    query_embedding = (
        model.encode(query, convert_to_tensor=True, show_progress_bar=False)
        .cpu()
        .detach()
        .numpy()
    )
    query_embedding = query_embedding.reshape(1, -1)
    # Normalize the query embedding for cosine similarity
    faiss.normalize_L2(query_embedding)

    # Perform the search
    distances, indices = index.search(
        query_embedding, index.ntotal
    )  # Search all for sorting

    if most_similar:
        return indices[0][:k]  # Top-k most similar (smallest distance)
    else:
        return indices[0][-k:]  # Top-k most dissimilar


def topk_examples(
    model,
    index,
    query,
    corpus,
    abstract2pmid,
    data,
    k=3,
    most_similar=True,
):
    """
    Given a query (context sentence), returns the top k most similar contexts from the corpus.
    ------
    model : sentence embedding model
    index : faiss index for best abstract retrieval
    query : str (abstract)
    corpus : list of str (containing all abstracts)
    asbtract2pmid : dict (abstract pmid)
    data : dict (pmid : abstract, spans)
    k : int (number of nearest neighbors)
    most_similar : bool (if True, return the most similar, else return the least similar)
    """
    indices = knn_query(model, index, query, k, most_similar=most_similar)
    result_list = []
    pmids = []
    for i, idx in enumerate(indices):
        abstract = corpus[idx]
        # print("Nearest neighbor mention : ", NN_mention)
        if abstract not in abstract2pmid:
            continue
        pmid = abstract2pmid[abstract]
        answer = data[pmid]["output"]
        result_list.append(
            f"""In the following abstract:
            {abstract} \n
            This would be the correct answer :
            {answer}\n
            """
        )
        pmids.append(pmid)

    res = "\n".join(result_list)

    return res, pmids


def tagged_abstract(data_path, pmid):
    """
    Function to retrieve the tagged abstract from the .json file
    ----------
    data_path : str
        The path to the dataset.json file
    pmid : int
        The pmid of the article to retrieve the spans
    """
    with open(data_path, "r") as file:
        dataset = ujson.load(file)
    tagged_abstract = defaultdict(list)
    for article in dataset:
        if article["pmid"] == pmid:
            for span in article["spans"]:
                tagged_abstract[span["tag"]].append(span["text"])
    return tagged_abstract


def create_spans_with_surrounding_text(text, spans):
    """
    This function creates a new list of spans with the surrounding text (+- 1 word) of each span

    Parameters:
    ----------
    text : str
        The abstract text
    spans : list of dict
        The spans extracted from the text : start, end, label, tag and text
    """
    new_spans = []
    # Use regex to split the text into words, including punctuation as separate tokens
    words_with_indices = [
        (m.group(0), m.start(), m.end()) for m in re.finditer(r"\S+|\s+", text)
    ]

    def find_word_indices(span_start, span_end):
        start_word_index = 0
        end_word_index = 0
        for i, (word, start_idx, end_idx) in enumerate(words_with_indices):
            if start_idx <= span_start < end_idx:
                start_word_index = i
            if start_idx <= span_end <= end_idx:
                end_word_index = i
                break
        return start_word_index, end_word_index

    for span in spans:
        start = span["start"]
        end = span["end"]
        tag = span["tag"]
        span_text = span["text"]

        start_word_index, end_word_index = find_word_indices(start, end)

        # Define the surrounding words range
        surrounding_start_index = max(0, start_word_index - 2)
        surrounding_end_index = min(len(words_with_indices), end_word_index + 3)

        # Extract surrounding words
        surrounding_words = words_with_indices[
            surrounding_start_index:surrounding_end_index
        ]
        surrounding_text = "".join(word for word, _, _ in surrounding_words)

        new_span = {
            "tag": tag,
            "text": span_text,
            "surrounding": surrounding_text.strip(),
        }

        new_spans.append(new_span)

    return new_spans


def generate_examples(data, k=3, seed=42):
    """
    Randomly samples k examples from the data dictionary and formats them into a string.

    Parameters:
    - data: dict of dict, where each key is an ID and values have "input" and "output".
    - k: int, number of examples to sample.
    - seed: int, random seed for reproducibility.

    Returns:
    - A formatted string containing k examples.
    """
    random.seed(seed)  # Set seed for reproducibility
    sampled_ids = random.sample(list(data.keys()), min(k, len(data)))  # Sample k IDs

    examples = ""
    for pmid in sampled_ids:
        abstract = data[pmid]["input"]
        output = data[pmid]["output"]
        examples += f"""
        In the following abstract:
        {abstract} \n
        This would be the correct answer:
        {output}\n
        """
    return examples.strip()


def generate_prompt_text(
    abstract,
    extraction_prompts,
    topk_examples,
    format_spans,
    annotation_instructions,
    noise=False,
):
    """
    Generate prompt text based on the prompt version and reasoning flag.
    ----------------
    abstract : str (abstract to tag)
    extraction_prompts : str (prompts for extraction)
    format_spans : str (format of the spans)
    annotation_instructions : str (annotation instructions)
    topk_examples : str (top k examples of similar contexts)
    noise : bool (whether the examples are noisy)
    """

    if noise:
        with open("prompts/Prompt_noisy_data.txt", "r") as file:
            prompt = file.read()
    else:
        with open("prompts/Prompt_gold_data.txt", "r") as file:
            prompt = file.read()

    prompt = prompt.replace("{Extraction Prompts}", str(extraction_prompts))
    prompt = prompt.replace("{Abstract}", abstract)
    prompt = prompt.replace("{Format Spans}", format_spans)
    prompt = prompt.replace("{Annotation Instructions}", annotation_instructions)
    prompt = prompt.replace("{Top K Examples}", str(topk_examples))

    return prompt


### Test for reducing hallucinations for Llama
""" 
Rules:
- Do NOT rewrite, summarize, or rephrase the text.
- Keep the text exactly as provided (including line breaks, typos, and formatting).
- The ONLY change you are allowed to make is to insert annotations in the format @@mention##ENTITY@@ into the original text.
- The output must be identical to the input text, except for these inserted annotations.
- Annotate both title and abstract.
"""


def prompt_vllm(system_instructions, llm, tokenizer, sampling_params, prompt):
    """
    system_instructions : str (instructions for the LLM)
    llm : LLM model
    tokenizer : AutoTokenizer
    sampling_params : SamplingParams config
    prompt : str (prompt text)
    """
    messages = [
        {"role": "system", "content": system_instructions},
        {"role": "user", "content": prompt},
    ]
    prompts = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    prompts = TokensPrompt(prompt_token_ids=prompts)  # Used for Mistrall-3.1 only
    # Decode the generated tokens into text
    outputs = llm.generate(prompts=prompts, sampling_params=sampling_params)
    answer = outputs[0].outputs[0].text

    return answer


def prompt_llm_hf(system_instructions, llm, tokenizer, prompt):
    """
    system_instructions : str (instructions for the LLM)
    llm : Hugging Face model (e.g., AutoModelForCausalLM)
    tokenizer : Hugging Face tokenizer (e.g., AutoTokenizer)
    prompt : str (prompt text)
    """
    # Format the input as a chat prompt
    messages = [
        {"role": "system", "content": system_instructions},
        {"role": "user", "content": prompt},
    ]
    # Use tokenizer to format the chat template
    chat_input = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # Tokenize the input for the Hugging Face model
    model_inputs = tokenizer(chat_input, return_tensors="pt").to(llm.device)

    # Generate output tokens using the model
    generated_ids = llm.generate(
        **model_inputs, no_repeat_ngram_size=35, max_new_tokens=8192
    )

    # Decode the generated tokens into text
    answer = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    return answer


def prompt_gpt(system_instructions, gpt_version, prompt, reasoning_effort="medium"):
    """
    system_instructions : str (instructions for the LLM)
    prompt : str (prompt text)
    gpt_version : Version of the GPT model
    """
    if not system_instructions:
        raise ValueError("Error: system_instructions cannot be None or empty.")
    if not prompt:
        raise ValueError("Error: prompt cannot be None or empty.")
    kwargs = {
        "model": gpt_version,
        "messages": [
            {"role": "system", "content": system_instructions},
            {"role": "user", "content": prompt},
        ],
    }

    if gpt_version in ["o3-mini"]:
        kwargs["reasoning_effort"] = reasoning_effort
    else:
        kwargs.update({"max_completion_tokens": 2048, "temperature": 0})

    completion = openai.chat.completions.create(**kwargs)

    return completion.choices[0].message.content


def gather_tagged_entities(original_text, llm_output, tag2label):
    """
    1) Extract tagged segments from `llm_output` into a `plain_text`,
       keeping track of entity offsets in that plain-text space.
    2) Use SequenceMatcher to align `plain_text` to `original_text`.
       We build a new `aligned_text` that favors the `original_text`
       whenever there's a mismatch (i.e. we fix the LLM's small errors).
    3) Remap entity offsets from the `plain_text` indices to their
       corresponding positions in the final `aligned_text`.

    Returns:
        entity_positions (list):
            Each entry has {start, end, label, tag, text} in *aligned_text* coords.
            If an entity cannot be aligned (fully deleted), start/end = -1.
        aligned_text (str):
            The merged text that is (virtually) the same as `original_text`
            wherever there were mismatches.
        count (int):
            Number of tagged entities found in `llm_output`.
    """

    # ----------------------------------------------------------------
    # 1) Extract entities & build plain_text (LLM output without tags)
    # ----------------------------------------------------------------
    pattern = r"@@(.*?)##(.*?)@@"
    entity_positions = []
    plain_text_parts = []
    current_idx = 0
    count = 0

    for match in re.finditer(pattern, llm_output):
        count += 1
        tagged_text, entity_type = match.groups()

        # Add preceding raw text
        plain_text_parts.append(llm_output[current_idx : match.start()])

        # Record entity offsets in the plain_text
        start_in_plain = sum(len(x) for x in plain_text_parts)
        end_in_plain = start_in_plain + len(tagged_text)
        if (
            entity_type not in tag2label
        ):  # Avoid hallucinations such as 1) "@@administrative## reasons; 123 patients from @@ reasons; 123 patients from" or 2) nonexistant tag (i.e. "T116")
            continue
        entity_positions.append(
            {
                "start": start_in_plain,
                "end": end_in_plain,
                "label": tag2label.get(entity_type, entity_type),
                "tag": entity_type,
                "text": tagged_text,
            }
        )

        # Add the actual entity text
        plain_text_parts.append(tagged_text)

        # Move past the tag in llm_output
        current_idx = match.end()

    # Append any leftover text after the last tag
    plain_text_parts.append(llm_output[current_idx:])
    plain_text = "".join(plain_text_parts)

    # ----------------------------------------------------------------
    # 2) Use SequenceMatcher to align plain_text -> original_text,
    #    building an "aligned_text" that heavily favors original_text.
    # ----------------------------------------------------------------

    matcher = SequenceMatcher(None, plain_text, original_text)

    aligned_text_parts = []
    # Map each plain_text index -> index in final aligned_text
    plain_to_aligned = [-1] * len(plain_text)

    aligned_idx = 0

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        # i1..i2 => slice of plain_text
        # j1..j2 => slice of original_text
        if tag == "equal":
            # The substrings are identical, so we keep them as-is from plain_text
            chunk = plain_text[i1:i2]
            aligned_text_parts.append(chunk)

            # Fill in the mapping
            for p in range(i1, i2):
                plain_to_aligned[p] = aligned_idx + (p - i1)

            aligned_idx += len(chunk)

        elif tag == "replace":
            # The LLM changed something. We trust original_text more:
            chunk = original_text[j1:j2]
            aligned_text_parts.append(chunk)

            # For replaced chunk in plain_text, we forcibly map them all
            # into the new substring in aligned_text (even though it's not
            # literally the LLM string). This ensures the final text
            # matches original_text for that region.
            length_replaced = i2 - i1
            length_chunk = len(chunk)
            for offset, p in enumerate(range(i1, i2)):
                # We clamp offset to the chunk length
                if offset >= length_chunk:
                    # Any leftover from plain_text is basically "lost"
                    plain_to_aligned[p] = -1
                else:
                    plain_to_aligned[p] = aligned_idx + offset

            aligned_idx += len(chunk)

        elif tag == "delete":
            # plain_text has extra text that doesn't appear in original_text,
            # so we skip that chunk entirely
            for p in range(i1, i2):
                plain_to_aligned[p] = -1
            # no text appended => aligned_idx doesn't move

        elif tag == "insert":
            # original_text has an extra piece that doesn't exist in plain_text
            # we insert it to make sure the final text matches original_text
            chunk = original_text[j1:j2]
            aligned_text_parts.append(chunk)
            aligned_idx += len(chunk)
            # We do NOT map anything from plain_text => this chunk

    aligned_text = "".join(aligned_text_parts)

    # ----------------------------------------------------------------
    # 3) Convert entity offsets (plain_text coords) -> aligned_text coords
    # ----------------------------------------------------------------
    for ent in entity_positions:
        start_pt = ent["start"]
        end_pt = ent["end"]

        # Collect all mapped indices from [start_pt..end_pt)
        mapped_idxs = []
        for pos in range(start_pt, end_pt):
            if 0 <= pos < len(plain_to_aligned):
                mapped = plain_to_aligned[pos]
                if mapped != -1:
                    mapped_idxs.append(mapped)

        if mapped_idxs:
            ent["start"] = min(mapped_idxs)
            ent["end"] = max(mapped_idxs) + 1
        else:
            # Nothing got aligned => treat as lost
            ent["start"] = -1
            ent["end"] = -1

    return entity_positions, aligned_text, count


def convert_to_seqeval_format(y_true, y_pred, abstract):
    """
    This function converts the entity data to seqeval format
    Parameters:
    ----------
    y_true : list
        Human annotated data
    y_pred : list
        Model predictions
    abstract : str
        The abstract to convert to seqeval format
    """

    def label_tokens(annotations, abstract):
        tokens = abstract.split()
        labels = ["O"] * len(tokens)
        for ann in annotations:
            start_idx = len(abstract[: ann["start"]].split())
            end_idx = start_idx + len(ann["text"].split())
            labels[start_idx] = f"B-{ann['tag']}"
            for i in range(start_idx + 1, end_idx):
                labels[i] = f"I-{ann['tag']}"
        return labels

    y_true_seqeval = label_tokens(y_true, abstract)
    y_pred_seqeval = label_tokens(y_pred, abstract)
    return y_true_seqeval, y_pred_seqeval


def label_tokens_from_offsets(text, annotations):
    doc = nlp(text)
    tokens = [token.text for token in doc]
    labels = ["O"] * len(tokens)

    for ann in annotations:
        start_char = ann["start"]
        end_char = ann["end"]
        start_token = next(
            (i for i, token in enumerate(doc) if token.idx >= start_char), None
        )
        end_token = next(
            (i for i, token in enumerate(doc) if token.idx >= end_char), None
        )

        if start_token is not None and end_token is not None:
            labels[start_token] = f"B-{ann['tag']}"
            for i in range(start_token + 1, end_token):
                labels[i] = f"I-{ann['tag']}"

    return labels


def compute_metrics(pmids, data, tag_to_label, pmid2pred_spans, debug=False):
    all_y_true_seqeval = []
    all_y_pred_seqeval = []
    results_list = []
    empty = 0

    for i, pmid in enumerate(pmids):
        abstract, true_spans = data[pmid]["input"], data[pmid]["spans"]

        pred_spans = pmid2pred_spans[pmid]

        if not pred_spans:
            print(f"Skipped pmid: {pmid} due to empty pred_spans")
            empty += 1
            continue

        y_true_seqeval = label_tokens_from_offsets(
            text=abstract, annotations=true_spans
        )
        y_pred_seqeval = label_tokens_from_offsets(
            text=abstract, annotations=pred_spans
        )
        all_y_true_seqeval.append(y_true_seqeval)
        all_y_pred_seqeval.append(y_pred_seqeval)
        results = seqeval.compute(
            predictions=[y_pred_seqeval], references=[y_true_seqeval]
        )
        if debug:
            print("index :", i, "pmid :", pmid)
            print("precision :", results["overall_precision"])
            print("recall :", results["overall_recall"])
            print("f1 :", results["overall_f1"])
            print("accuracy :", results["overall_accuracy"])
        results["pmid"] = pmid
        results_list.append(results)

        if i % 100 == 0:
            print("i :", i)

    nb_valid_eval = len(results_list)
    # Evaluate using seqeval
    overall_results = seqeval.compute(
        predictions=all_y_pred_seqeval, references=all_y_true_seqeval
    )
    overall_class_specific_f1 = {
        k: v["f1"] for k, v in overall_results.items() if not k.startswith("overall")
    }

    df_results = pd.DataFrame(results_list)
    columns = [
        "pmid",
        "overall_accuracy",
        "overall_precision",
        "overall_recall",
        "overall_f1",
    ]  # + \
    # [col for col in df_results.columns if col not in ['pmid', 'overall_accuracy', 'overall_precision', 'overall_recall', 'overall_f1']]
    df_results = df_results[columns].rename(
        columns={
            "overall_accuracy": "accuracy",
            "overall_precision": "precision",
            "overall_recall": "recall",
            "overall_f1": "f1",
        }
    )

    return (
        {
            "accuracy": overall_results["overall_accuracy"],
            "precision": overall_results["overall_precision"],
            "recall": overall_results["overall_recall"],
            "f1": overall_results["overall_f1"],
            "class_specific_f1": overall_class_specific_f1,
            "detailed_results": overall_results,
        },
        df_results,
        nb_valid_eval,
        empty,
    )


def escape_newlines_inside_json(s):
    """
    Escapes literal newline characters inside JSON string literals.
    It scans the input character-by-character, and whenever it is inside a
    double-quoted string, it replaces actual newline characters with the two-character
    sequence '\\n'.
    """
    in_string = False
    escaped = False
    result = []
    for char in s:
        if char == '"' and not escaped:
            in_string = not in_string
        if char == "\n" and in_string:
            result.append("\\n")
        else:
            result.append(char)
        # Update escaped status
        if char == "\\" and not escaped:
            escaped = True
        else:
            escaped = False
    return "".join(result)


def parse_answer(answer):
    """
    Function to parse the answer as JSON, handling mismatched quotes, unescaped characters,
    smart quotes, trailing commas, and more.
    """
    # Clean the answer string by stripping unwanted artifacts
    answer = answer.strip("```json").strip("```").strip()
    answer = escape_newlines_inside_json(answer)
    try:
        # Attempt to parse as JSON directly
        return json.loads(answer)
    except json.JSONDecodeError:
        # Start cleaning known problematic patterns
        try:
            # Normalize smart quotes to standard quotes
            answer = re.sub(r"[“”]", '"', answer)

            # Fix mismatched or malformed quotes
            answer = re.sub(
                r"(?<!\\)'", '"', answer
            )  # Replace single quotes with double quotes

            # Fix keys with single quotes
            answer = re.sub(r"(?<=[{,])\s*'([a-zA-Z0-9_]+)'\s*:", r'"\1":', answer)

            # Remove invalid control characters
            answer = re.sub(r"[\t\r]", "", answer)

            # Remove trailing commas
            answer = re.sub(r",\s*([}\]])", r"\1", answer)

            # Fix unescaped Unicode characters (e.g., \u201c)
            answer = re.sub(
                r"\\u([0-9a-fA-F]{4})", lambda m: chr(int(m.group(1), 16)), answer
            )

            # Attempt to parse again
            return json.loads(answer)
        except json.JSONDecodeError as e:
            # Log the remaining problematic string for debugging
            print("Failed to parse cleaned JSON:", str(e))
            # print("Problematic JSON snippet:", answer)
            return None


def extract_final_output(answer):
    """
    This is specific to the reasoning model that generates too much text so
    Extracts the last valid block of JSON code in the format ```json ... ``` from the answer.
    If the last block is incomplete, it tries to find the second-to-last block. (due to lack of tokens but usually it already found a good answer even before)
    ----------
    answer : str
        The full response from the LLM.
    Returns:
        The cleaned JSON block or None if no valid block is found.
    """
    # Find all occurrences of ```json
    json_blocks = [match.start() for match in re.finditer(r"```json", answer)]

    if not json_blocks:
        print("No JSON block starting with ```json found.")
        return None

    # Traverse JSON blocks from the last to the first
    for start_pos in reversed(json_blocks):
        # Find the next closing ```
        json_end = answer.find("```", start_pos + 7)
        if json_end != -1:
            # Extract the block including the ```json
            return answer[start_pos : json_end + 3].strip()

    print("No valid JSON block with closing ``` found.")
    return None


def extract_last_assistant_message(output):
    """
    Extract the last assistant message from a structured output.
    ----------
    - output : str
    The generated text including system, user, and assistant messages.
    Returns: (str) The last assistant's message.
    """
    # Split the output by role markers (system, user, assistant or model)
    chunks = re.split(r"\n(?:assistant|model)\n", output)
    # The last chunk is the assistant's final message
    if len(chunks) > 1:
        return chunks[-1].strip()  # Return the last assistant's message
    else:
        return output.strip()  # Fallback if no "assistant" marker exists


def extract_json_from_text(text):
    """
    Extract the JSON portion from a given text.
    ----------
    - text : str
      The input text containing a JSON block.
    Returns: (str) The JSON block if found, else an empty string.
    """
    # Use a regex pattern to match the JSON block
    json_match = re.search(r"```json\n(.*?)\n```", text, re.DOTALL)
    if json_match:
        return json_match.group(1).strip()  # Return the JSON block
    else:
        return ""  # Return empty string if no JSON block is found


def extract_last_json_from_text(text):
    """
    Extract the last JSON block from a given text enclosed in triple backticks and labeled as json.
    ----------
    - text : str
      The input text containing one or more ```json ... ``` blocks.
    Returns: (str) The last JSON block if found, else an empty string.
    """
    # Find all JSON blocks formatted as ```json\n...\n```
    json_blocks = re.findall(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    if json_blocks:
        return json_blocks[-1].strip()
    else:
        return ""


def create_tagged_text(abstract, pred_spans):
    """
    Function which modifies the abstract to include the entity:
    Input : "DCTN4 as a modifier of chronic Pseudomonas aeruginosa infection"
    Output : "@@DCTN4##T103@@ as a modifier of @@chronic Pseudomonas aeruginosa infection##T038@@"
    If two or more spans overlap, only the span with the earliest start is kept.
    -------------
    Parameters:
    - abstract (str): The original text
    - pred_spans (list of dict): A list of spans with entity details (start, end, text, tag)

    Returns:
    - new_abstract (str): The text modified with entity annotations
    """
    # First, sort spans by start (and end if needed) in ascending order.
    sorted_spans = sorted(pred_spans, key=lambda x: (x["start"], x["end"]))

    # Build list with non overlapping spans: if a span overlaps any previously accepted one,
    # we keep only the one with the earliest start (already accepted).
    allowed_spans = []
    current_end = -1  # initial value before any span is accepted
    for span in sorted_spans:
        if span["start"] >= current_end:
            allowed_spans.append(span)
            current_end = span["end"]

    # For text replacement, sort allowed spans in reverse order by start.
    allowed_spans = sorted(allowed_spans, key=lambda x: x["start"], reverse=True)
    new_abstract = abstract

    # Replace text segments with the tagged annotation.
    for span in allowed_spans:
        start = span["start"]
        end = span["end"]
        text = span["text"]
        entity_tag = span["tag"]
        tagged_text = f"@@{text}##{entity_tag}@@"
        new_abstract = new_abstract[:start] + tagged_text + new_abstract[end:]

    return new_abstract


def fuse_adjacent_tag(true_spans):
    """
    Merge adjacent true spans that have the same label and tag.
    """
    if len(true_spans) == 0:
        return []
    new_true_spans = []
    prev = true_spans[0]
    for t in true_spans[1:]:
        if prev["tag"] == t["tag"] and prev["end"] + 1 == t["start"]:
            prev = {
                "start": prev["start"],
                "end": t["end"],
                "label": prev["label"],
                "tag": prev["tag"],
                "text": prev["text"] + " " + t["text"],
            }
        else:
            new_true_spans.append(prev)
            prev = t
    new_true_spans.append(prev)
    return new_true_spans


################################################################################################################
########################### FOR DATASET CORRUPTION PART (dataset_corruption2.ipynb) ############################
################################################################################################################


def generate_prompt_dataset_corruption(
    abstract, annotated_abstract, extraction_prompts, format_spans, noise_instructions
):
    """
    Generate prompt text based on the prompt version and reasoning flag.
    ----------------
    abstract : str (abstract to tag)
    extraction_prompts : str (prompts for extraction)
    format_spans : str (format of the spans)
    noise_instructions : str (instructions for encouraging a certain level of noise)
    """

    prompt = f"""
    A professional medical doctor annotated a research article abstract.
    He labeled all text that fell in of these categories: \n {extraction_prompts}

    This is the original abstract without tags:
    {abstract}

    This is the annotated abstract with tags:
    {annotated_abstract}

    Your task is to review the annotated abstract and correct any mistakes.

    Errors can appear in many forms :
    - Incorrect Label: If a label is wrong, replace it with the correct one. Example: '@@aspirin##Food@@' → '@@aspirin##Drug@@'
    - Wrong overlap: If the selected text is too long, remove any extra words while keeping the annotation precise or/and if the tagged text is too short and lacks key details, expand it to include the necessary information.. Example: '@@severe chronic pain##Symptom@@' → '@@chronic pain##Symptom@@'. Example: '@@heart##Anatomical_Structure@@' → '@@heart muscle##Anatomical_Structure@@'
    - Multiple Entities in One Tag: If a single tag contains multiple entities, split them into separate tags. Example: '@@blood pressure medication##Drug@@' → '@@blood pressure##Measurement@@ @@medication##Drug@@'
    - Invalid Tag: If text was tagged incorrectly and should not have been annotated, remove the tag. Example: '@@water##Drug@@' → 'water' (no annotation)
    - Missing Spans: If the model forgot to tag a text. Example: 'The heart is a vital organ.' → 'The @@heart##Anatomical_Structure@@ is a vital organ.'

    All text were tagged in this format : {format_spans}
    It's very important that you keep the same structure for all the corrections you made.

    {noise_instructions}
    """
    prompt += """
    Return the answer in a JSON format.
    An example of valid answer look like this :
        **Final Output**
        ```json
        {"output": text ... @@mentions##entity@@ ... text}
        ```
    """

    return prompt


def compute_noise(pred_spans, true_spans, debug=False):
    """
    Two-loop approach with a separate missing loop.

    Loop 1: Identify correct predictions.
      - A predicted span is marked correct only if its start, end, label, and tag exactly match a true span.
      - In that case, the true span is marked as claimed.

    Loop 2: Process remaining predictions.
      - For each non-correct prediction, we gather all true spans that overlap it.
      - All overlapping true spans are marked as claimed.
      - If no overlap is found, error_type = "invalid_tag".
      - If multiple overlaps are found, error_type = "multiple_entities".
      - If exactly one overlap is found:
            If the boundaries match exactly (but the prediction wasn’t marked correct), error_type = "wrong_label".
            Otherwise, error_type = "wrong_overlap".
      - However, if the tag and label match, the prediction and true span start at the same position, their end positions differ by exactly one,
        and the only difference in text is that one has an extra "s" (i.e. either candidate["text"] == pred["text"] + "s"
        or vice versa), then the prediction is treated as correct.

    Loop 3: For each true span, if no prediction overlaps it (i.e. it was never claimed), add an error with error_type "missing".

    Returns:
      f1 (float): F1 score.
      precision (float): Precision.
      recall (float): Recall.
      correct_count (int): Number of correct predictions.
      noise_count (int): Number of erroneous predictions.
      results (list): List of dicts for each predicted (or missing) span with keys:
                      'start', 'end', 'label', 'tag', 'text', 'correct', and 'error_type'.
      noise_types (dict): Counts for each error type.
    """
    noise_types = {
        "wrong_label": 0,
        "wrong_overlap": 0,
        "multiple_entities": 0,
        "invalid_tag": 0,
        "missing": 0,
    }
    results = []
    correct_count = 0
    noise_count = 0
    n_pred = len(pred_spans)
    n_true = len(true_spans)

    # claimed_true: set of indices of true spans that are overlapped by any prediction.
    claimed_true = set()
    # correct_preds: set of indices of predictions that are exact matches.
    correct_preds = set()

    # Loop 1: Identify correct predictions (exact match)
    for i, p in enumerate(pred_spans):
        matched = False
        for j, t in enumerate(true_spans):
            if (
                p["start"] == t["start"]
                and p["end"] == t["end"]
                and p["label"] == t["label"]
                and p["tag"] == t["tag"]
            ):
                results.append(
                    {
                        "start": p["start"],
                        "end": p["end"],
                        "label": p["label"],
                        "tag": p["tag"],
                        "text": p["text"],
                        "correct": True,
                        "error_type": None,
                    }
                )
                correct_count += 1
                correct_preds.add(i)
                claimed_true.add(j)
                matched = True
                if debug:
                    print(f"Loop1: Prediction {i} is correct (matches true span {j}).")
                break
        if not matched and debug:
            print(f"Loop1: Prediction {i} is not correct.")

    # Loop 2: Process remaining predictions (errors)
    for i, p in enumerate(pred_spans):
        if i in correct_preds:
            continue  # Skip already correct predictions.
        # Find all true spans that overlap the prediction.
        overlaps = []
        for j, t in enumerate(true_spans):
            if p["start"] < t["end"] and p["end"] > t["start"]:
                overlaps.append((j, t))
        # Mark every overlapping true span as claimed.
        for j, t in overlaps:
            claimed_true.add(j)
        if not overlaps:
            error_type = "invalid_tag"
            noise_types[error_type] += 1
            noise_count += 1
            results.append(
                {
                    "start": p["start"],
                    "end": p["end"],
                    "label": p["label"],
                    "tag": p["tag"],
                    "text": p["text"],
                    "correct": False,
                    "error_type": error_type,
                }
            )
            if debug:
                print(f"Loop2: Prediction {i} => {error_type} (no overlaps).")
            continue
        if len(overlaps) > 1:
            error_type = "multiple_entities"
            noise_types[error_type] += 1
            noise_count += 1
            results.append(
                {
                    "start": p["start"],
                    "end": p["end"],
                    "label": p["label"],
                    "tag": p["tag"],
                    "text": p["text"],
                    "correct": False,
                    "error_type": error_type,
                }
            )
            if debug:
                print(f"Loop2: Prediction {i} => {error_type} (multiple overlaps).")
            continue
        # Exactly one overlapping true span.
        cand_idx, candidate = overlaps[0]

        # Special case: if tag and label match, spans start at the same index, and the difference in end indices is 1,
        # and the texts differ only by an extra 's' (i.e. candidate["text"] equals p["text"] + "s" or vice versa),
        # then treat the prediction as correct.
        if (
            p["tag"] == candidate["tag"]
            and p["label"] == candidate["label"]
            and p["start"] == candidate["start"]
            and abs(p["end"] - candidate["end"]) == 1
            and (
                candidate["text"] == p["text"] + "s"
                or p["text"] == candidate["text"] + "s"
            )
        ):
            results.append(
                {
                    "start": p["start"],
                    "end": p["end"],
                    "label": p["label"],
                    "tag": p["tag"],
                    "text": p["text"],
                    "correct": True,
                    "error_type": None,
                }
            )
            correct_count += 1
            if debug:
                print(
                    f"Loop2: Prediction {i} => special correct (plural 's' adjustment)."
                )
        else:
            if p["start"] == candidate["start"] and p["end"] == candidate["end"]:
                # Boundaries match exactly but label or tag differ.
                error_type = "wrong_label"
                noise_types[error_type] += 1
                noise_count += 1
                results.append(
                    {
                        "start": p["start"],
                        "end": p["end"],
                        "label": p["label"],
                        "tag": p["tag"],
                        "text": p["text"],
                        "correct": False,
                        "error_type": error_type,
                    }
                )
                if debug:
                    print(
                        f"Loop2: Prediction {i} => {error_type} (exact boundaries but label/tag mismatch)."
                    )
            else:
                error_type = "wrong_overlap"
                noise_types[error_type] += 1
                noise_count += 1
                results.append(
                    {
                        "start": p["start"],
                        "end": p["end"],
                        "label": p["label"],
                        "tag": p["tag"],
                        "text": p["text"],
                        "correct": False,
                        "error_type": error_type,
                    }
                )
                if debug:
                    print(f"Loop2: Prediction {i} => {error_type} (partial overlap).")

    # Loop 3: Add missing errors for true spans with no overlapping prediction
    for j, t in enumerate(true_spans):
        # A true span is considered covered if any prediction overlaps it.
        overlap_found = any(
            p["start"] < t["end"] and p["end"] > t["start"] for p in pred_spans
        )
        if not overlap_found:
            results.append(
                {
                    "start": t["start"],
                    "end": t["end"],
                    "label": t["label"],
                    "tag": t["tag"],
                    "text": t["text"],
                    "correct": False,
                    "error_type": "missing",
                }
            )
            noise_types["missing"] += 1
            if debug:
                print(f"Loop3: True span {j} => missing (no prediction overlaps).")

    results = sorted(results, key=lambda x: (x["start"], x["end"]))
    precision = correct_count / n_pred if n_pred > 0 else 0
    recall = correct_count / n_true if n_true > 0 else 0
    f1 = (
        2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    )
    return f1, precision, recall, correct_count, noise_count, results, noise_types


def find_exact_matching(spans, start, end):
    # Find t in true_spans such that t["start"] == start and t["end"] == end
    # This means that the pred_span is exactly matching the span
    return next((d for d in spans if d["start"] == start and d["end"] == end), None)


def find_matching_multiple_entities(true_spans, start, end):
    """
    Return all gold-standard spans (t in true_spans)
    that overlap with the predicted range [start, end].
    Overlap means they share at least one character.
    """
    matched = []
    for t in true_spans:
        # Two intervals [start, end] and [t["start"], t["end"]] overlap
        # if the start of one is not beyond the end of the other:
        if not (t["end"] < start or t["start"] > end):
            matched.append(t)
    return matched


def find_wrong_overlap(true_spans, pred_start, pred_end, min_overlap=1):
    """
    Find the true span that has the maximum overlap with the predicted span
    identified as a wrong_overlap error. Returns a tuple (error_type, true_span)
    where true_span is the candidate correction if the overlap is at least min_overlap.
    """
    best_overlap = 0
    best_span = None
    for t in true_spans:
        # Calculate the amount of overlap between the predicted span and the true span.
        overlap = min(pred_end, t["end"]) - max(pred_start, t["start"])
        if overlap > best_overlap and overlap >= min_overlap:
            best_overlap = overlap
            best_span = t
    return ("wrong_overlap", best_span)


def reduce_noise_per_abstract(
    correct_count,
    noise_count,
    pred_spans,
    true_spans,
    reduction,
    results,
    debug=False,
    seed=42,
):
    """
    Function to reduce the noise of the predicted spans based on the true_spans.
    ----------
    - f1 : float (f1 score)
    - precision : float (precision score)
    - recall : float (recall score)
    - correct_count : int (number of correct annotations)
    - noise_count : int (number of incorrect annotations)
    - pred_spans / true_spans : list of dicts (predicted spans / true spans)
        - start: start index of the span
        - end: end index of the span
        - label: label of the span
        - tag: tag of the span
        - text: text of the span
    - reduction : float (By how much do we reduce the noise).
        For instance if the original noise is 30% and we set reduction to 0.5, the new noise will be 15%
    - results : list of dicts (results of the evaluation)
        - start: start index of the span
        - end: end index of the span
        - label: label of the span
        - tag: tag of the span
        - text: text of the span
        - correct: boolean (is the span correct or not)
        - error_type: type of error (invalid_tag, wrong_label, missing, too much, too little, multiple_entities)
    ----------
    Returns:
        - new_spans : list of dicts (new predicted spans)
    """
    n_pred = len(pred_spans)
    n_true = len(true_spans)
    if debug:
        print(f"n_pred : {n_pred} | Correct: {correct_count} | Wrong: {noise_count}")
    precision = correct_count / n_pred if n_pred > 0 else 0
    recall = correct_count / n_true if n_true > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    noise = 1 - f1

    noise_goal = noise * (1 - reduction)

    new_spans = []
    # Add all "correct" spans to new_spans before shuffling
    for result in results:
        if result["correct"]:
            correct_span = find_exact_matching(
                pred_spans, result["start"], result["end"]
            )
            if correct_span not in new_spans:
                new_spans.append(correct_span)

    # Filter out the "correct" results so they won't be processed again.
    non_correct_results = [res for res in results if not res["correct"]]

    # Shuffle only the non-correct results
    random.seed(seed)
    random.shuffle(non_correct_results)

    for idx, result in enumerate(non_correct_results):
        # Process only non-correct results
        if noise > noise_goal:
            if result["error_type"] == "invalid_tag":
                n_pred -= 1
                noise_count -= 1
            elif result["error_type"] == "wrong_label":
                correct_span = find_exact_matching(
                    true_spans, result["start"], result["end"]
                )
                if correct_span is not None and correct_span not in new_spans:
                    new_spans.append(correct_span)
                    correct_count += 1
                    noise_count -= 1
            elif result["error_type"] == "missing":
                correct_span = find_exact_matching(
                    true_spans, result["start"], result["end"]
                )
                if correct_span is not None and correct_span not in new_spans:
                    new_spans.append(correct_span)
                    n_pred += 1
                    correct_count += 1
            elif result["error_type"] == "wrong_overlap":
                error_type_name, correct_span = find_wrong_overlap(
                    true_spans, result["start"], result["end"]
                )
                if correct_span is not None and correct_span not in new_spans:
                    new_spans.append(correct_span)
                    correct_count += 1
                    noise_count -= 1
            elif result["error_type"] == "multiple_entities":
                matches = find_matching_multiple_entities(
                    true_spans, result["start"], result["end"]
                )
                if debug:
                    print(f"multiple entities: {len(matches)}")
                for j, match in enumerate(matches):
                    if match and match not in new_spans:
                        new_spans.append(match)
                        correct_count += 1
                        if j == 0:
                            noise_count -= 1
                        if j >= 1:
                            n_pred += 1
            else:
                # For any unknown error type, keep the original predicted span
                print("Error: unknown error type")
                correct_span = find_exact_matching(
                    pred_spans, result["start"], result["end"]
                )
                if correct_span is not None and correct_span not in new_spans:
                    new_spans.append(correct_span)
        else:
            # if noise threshold reached, use the predicted span as a fallback
            predicted_span = find_exact_matching(
                pred_spans, result["start"], result["end"]
            )
            if predicted_span is not None and predicted_span not in new_spans:
                new_spans.append(predicted_span)

        # Recalculate metrics after processing each non-correct result
        precision = correct_count / n_pred if n_pred > 0 else 0
        recall = correct_count / n_true if n_true > 0 else 0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall > 0
            else 0
        )
        noise = 1 - f1

        if debug:
            see = {k: result[k] for k in ["start", "end", "label", "tag", "text"]}
            print(
                f"Index: {idx} | n_pred: {n_pred} | Correct: {correct_count} | Wrong: {noise_count} | {result['error_type']} | pred_spans: {see}"
            )

    new_spans = sorted(new_spans, key=lambda x: (x["start"], x["end"]))
    if debug:
        print(
            f"\nNoise: {noise} | Correct: {correct_count} | Wrong: {noise_count} | n_true: {len(true_spans)} \n"
            f"f1: {f1} | precision: {precision} | recall: {recall}"
        )

    return new_spans


def reduce_noise(
    full_results,
    wanted_noise,
    reduction,
    total_n_true,
    debug=False,
):
    """
    Function to reduce the noise of the dataset to final_noise.
    ----------
    - full_results : list of dicts (results of the evaluation)
        - noise : float (noise of the abstract)
        - noise_count : int (number of noisy spans introduced during the corruption)
        - n_true : int (number of spans in true_spans)
        - pred_spans / true_spans : list of dicts (predicted spans / true spans)
            - start: start index of the span
            - end: end index of the span
            - label: label of the span
            - tag: tag of the span
            - text: text of the span
        - pmid : str (pmid of the abstract)
        - results : list of dicts (results of the evaluation)
            - start: start index of the span
            - end: end index of the span
            - label: label of the span
            - tag: tag of the span
            - text: text of the span
            - correct: boolean (is the span correct or not)
            - error_type: type of error (invalid_tag, wrong_label, missing, too much, too little, multiple_entities)
    - wanted_noise : float (final noise we want to reach)
        For instance if the original noise over the whole dataset is 30% and we set "final_noise" to 0.1, the final noise over the whole dataset will be 20%
    - reduction : float (By how much do we reduce the noise in one abstract).
        For instance if the original noise in one abstract is 30% and we set reduction to 0.5, the new noise on this abstract will be 15%
    - total_n_true : int (total number of true spans)
    ----------
    Returns:
        - new_full_results : list of dicts (similar to full_results but with some corrections from true_spans in order to reduce the noise)
    """

    current_noise_count = sum([item["noise_count"] for item in full_results])
    current_correct_count = sum([item["correct_count"] for item in full_results])

    print("Initial correct counts :", current_correct_count)
    print("Initial noise counts :", current_noise_count)

    precision = current_correct_count / (current_correct_count + current_noise_count)
    recall = current_correct_count / total_n_true
    f1 = 2 * precision * recall / (precision + recall)
    current_noise = 1 - f1

    new_full_results = []
    random_indices = random.sample(full_results, len(full_results))
    idx = 0
    while current_noise > wanted_noise and idx < len(random_indices):
        random_res = full_results[idx]

        new_spans = reduce_noise_per_abstract(
            random_res["correct_count"],
            random_res["noise_count"],
            random_res["pred_spans"],
            random_res["true_spans"],
            reduction,
            random_res["results"],
        )  # new_spans has the exact same structure than pred_spans and true_spans
        (
            new_f1,
            new_precision,
            new_recall,
            new_correct_count,
            new_noise_count,
            new_results,
            noise_types,
        ) = compute_noise(new_spans, random_res["true_spans"])

        current_noise_count -= random_res["noise_count"] - new_noise_count
        current_correct_count -= random_res["correct_count"] - new_correct_count
        precision = current_correct_count / (
            current_correct_count + current_noise_count
        )
        recall = current_correct_count / total_n_true
        f1 = 2 * precision * recall / (precision + recall)
        current_noise = 1 - f1

        results = {
            "pmid": random_res["pmid"],
            "noise": 1 - new_f1,
            "f1": new_f1,
            "precision": new_precision,
            "recall": new_recall,
            "correct_count": new_correct_count,
            "noise_count": new_noise_count,
            "pred_spans": new_spans,
            "true_spans": random_res["true_spans"],
            "results": new_results,
        }
        new_full_results.append(results)
        if debug:
            print(
                "current_noise :",
                current_noise_count,
                "|| initial noise count :",
                random_res["noise_count"],
                "|| noise count after reduction :",
                new_noise_count,
                "|| initial correct count :",
                random_res["correct_count"],
                "|| correct count after reduction :",
                new_correct_count,
            )
        idx += 1
    print("Reached correct counts :", current_correct_count)
    print("Reached noise counts :", current_noise_count)
    print("Number of abstracts with noise reduced :", idx)

    while idx < len(random_indices):
        new_full_results.append(full_results[idx])
        idx += 1

    return new_full_results


def reduce_noise_per_abstract_distribution(
    correct_count,
    noise_count,
    pred_spans,
    true_spans,
    reduction,
    results,
    noise_types,
    errors_to_remove,
    debug=False,
    seed=42,
):
    """
    Similar to "reduce_noise_per_abstract" function but target a specific error type distribution.

    Parameters:
      - correct_count: int, number of correct annotations.
      - noise_count: int, number of incorrect annotations.
      - pred_spans / true_spans: list of dicts with keys: start, end, label, tag, text.
      - reduction: float, fraction by which the noise is reduced.
          For example, if the original noise is 30% and reduction is 0.5, the new noise will be 15%.
      - results: list of dicts (evaluation results) with keys: start, end, label, tag, text, correct, error_type.
      - noise_types: dict with counts per error type, e.g.
           {"wrong_label": 2, "wrong_overlap": 7, "multiple_entities": 0, "invalid_tag": 19, "missing": 8}
      - errors_to_remove: dict with target removals per error type, e.g.
           {"wrong_label": 0, "wrong_overlap": 9, "multiple_entities": 0, "invalid_tag": 5, "missing": 0}
      - debug: boolean flag to print debug information.
      - seed: int, seed for randomness.

    Returns:
      - new_spans: list of dicts, the updated predicted spans after noise reduction.
    """
    import random

    # Initialize counts from inputs.
    n_pred = len(pred_spans)
    n_true = len(true_spans)
    if debug:
        print(
            f"Initial n_pred: {n_pred} | Correct: {correct_count} | Noise: {noise_count}"
        )

    # Calculate precision, recall, F1, and define noise.
    precision = correct_count / n_pred if n_pred > 0 else 0
    recall = correct_count / n_true if n_true > 0 else 0
    f1 = (
        2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    )
    noise = 1 - f1

    # Set a noise goal by reducing the original noise by the given fraction.
    noise_goal = noise * (1 - reduction)

    # Determine the allowed number of removals per error type.
    allowed_removals = {}
    for err, target in errors_to_remove.items():
        if target > 0 and noise_types.get(err, 0) > 0:
            allowed_removals[err] = min(target, noise_types[err])

    # Track the number of errors fixed per error type.
    fixed_errors = {err: 0 for err in errors_to_remove.keys()}

    new_spans = []

    # Pre-add all "correct" spans from results to new_spans.
    for result in results:
        if result["correct"]:
            correct_span = find_exact_matching(
                pred_spans, result["start"], result["end"]
            )
            if correct_span is not None and correct_span not in new_spans:
                new_spans.append(correct_span)
            if debug:
                print(f"Added correct span at start: {result['start']}-{result['end']}")

    # Filter out the correct results so they are not processed again.
    non_correct_results = [res for res in results if not res["correct"]]

    # Shuffle the non-correct results so corrections occur in a random order.
    random.seed(seed)
    random.shuffle(non_correct_results)

    for idx, result in enumerate(non_correct_results):
        err_type = result["error_type"]
        # If the error type is targeted for removal and we haven't exceeded limits,
        # and if the current noise is still above our noise goal, then correct the error.
        if (
            err_type in allowed_removals
            and fixed_errors[err_type] < allowed_removals[err_type]
            and noise > noise_goal
        ):
            if err_type == "invalid_tag":
                if debug:
                    print(f"Index: {idx} | Correcting invalid_tag error")
                n_pred -= 1
                noise_count -= 1
                fixed_errors["invalid_tag"] += 1

            elif err_type == "wrong_label":
                correct_span = find_exact_matching(
                    true_spans, result["start"], result["end"]
                )
                if correct_span is not None and correct_span not in new_spans:
                    new_spans.append(correct_span)
                    correct_count += 1
                    noise_count -= 1
                fixed_errors["wrong_label"] += 1

            elif err_type == "missing":
                correct_span = find_exact_matching(
                    true_spans, result["start"], result["end"]
                )
                if correct_span is not None and correct_span not in new_spans:
                    new_spans.append(correct_span)
                    n_pred += 1
                    correct_count += 1
                fixed_errors["missing"] += 1

            elif err_type == "wrong_overlap":
                error_type_name, correct_span = find_wrong_overlap(
                    true_spans, result["start"], result["end"]
                )
                if correct_span is not None and correct_span not in new_spans:
                    new_spans.append(correct_span)
                    correct_count += 1
                    noise_count -= 1
                fixed_errors["wrong_overlap"] += 1

            elif err_type == "multiple_entities":
                if debug:
                    print(f"Index: {idx} | Correcting multiple_entities error")
                matches = find_matching_multiple_entities(
                    true_spans, result["start"], result["end"]
                )
                adds = 0
                noise_reduction = False
                for j, match in enumerate(matches):
                    if match is not None and match not in new_spans:
                        new_spans.append(match)
                        adds += 1
                        noise_reduction = True
                if noise_reduction:
                    noise_count -= 1
                n_pred += adds - 1
                correct_count += adds
                fixed_errors["multiple_entities"] += 1

            else:
                # If error type is unknown, fall back to using the predicted span.
                correct_span = find_exact_matching(
                    pred_spans, result["start"], result["end"]
                )
                if correct_span is not None and correct_span not in new_spans:
                    new_spans.append(correct_span)
        else:
            # If the error is not scheduled for correction or limits have been reached,
            # simply use the predicted span.
            span = find_exact_matching(pred_spans, result["start"], result["end"])
            if span is not None and span not in new_spans:
                new_spans.append(span)

        # Recompute metrics after processing each result.
        precision = correct_count / n_pred if n_pred > 0 else 0
        recall = correct_count / n_true if n_true > 0 else 0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0
        )
        noise = 1 - f1

        if debug:
            print(
                f"After index {idx} | n_pred: {n_pred} | Correct: {correct_count} | Noise: {noise_count} | "
                f"Precision: {precision:.3f} | Recall: {recall:.3f} | F1: {f1:.3f} | Noise metric: {noise:.3f}"
            )

    # # Remove all pred_span that correspond to the same true_span if any of these pred_span is corrected.
    # for span in new_spans:
    #     err_type = result["error_type"]
    #     if err_type == "wrong_overlap":
    #         error_type_name, correct_span = find_wrong_overlap(
    #             true_spans, result["start"], result["end"]
    #         )
    #         if correct_span in new_spans:
    #             new_spans.remove(span)

    new_spans = sorted(new_spans, key=lambda x: (x["start"], x["end"]))

    if debug:
        print(
            f"\nFinal metrics: n_pred: {n_pred} | Correct: {correct_count} | Noise: {noise_count} | n_true: {n_true}"
        )
        print(
            f"Final scores: Precision: {precision:.3f} | Recall: {recall:.3f} | F1: {f1:.3f} | Noise: {noise:.3f}"
        )

    return new_spans


def compute_noise_coefficient(
    wanted_noise: float,
    error_analysis: dict,
    initial_correct_counts: int,
    initial_noise_counts: int,
    n_true_spans: int,
    target_distribution: dict,
    debug=False,
) -> tuple:
    """
    Compute coefficient A such that scaling the target numbers by A yields the final noise level = wanted_noise.

    The final noise is defined as 1 - F1, where F1 is computed from adjusted correct and noise counts.

    New correction rules:
      - For correct counts:
          +1 for every correction of "wrong_overlap", "wrong_label", and "missing"
          +2 for every correction of "multiple_entities"
          (No correct increment for "invalid_tag")
      - For noise counts:
          -1 for every correction of "wrong_overlap", "wrong_label", "multiple_entities", and "invalid_tag"
          (No noise adjustment for "missing")

    This version uses a binary search (bisection) method to robustly converge on the value of A.

    Parameters:
      wanted_noise: Desired final noise level (e.g. 0.3 for 30% noise).
      error_analysis: A dict with error type counts.
      initial_correct_counts: Initial count of correct items.
      initial_noise_counts: Initial count of noise items.
      n_true_spans: Total number of true spans.
      target_distribution: Dict with target proportions per error type, e.g.,
                           {
                             'wrong_label': 0.15,
                             'wrong_overlap': 0.225,
                             'multiple_entities': 0.1,
                             'invalid_tag': 0.3,
                             'missing': 0.225
                           }
      debug: If True, prints intermediate values for debugging.

    Returns:
      A tuple: (A, final_noise, final_correct_count, final_noise_count, target_numbers_scaled)
             where target_numbers_scaled is a dict mapping error types to their scaled integer target counts.
    """
    # Precompute target numbers as floats; conversion to int is done in each iteration.
    target_numbers = {
        key: target_distribution[key] * wanted_noise * n_true_spans
        for key in target_distribution
    }

    def final_metrics(A: float):
        """
        For a given coefficient A, compute:
          - final_noise: 1 - F1,
          - final_correct_count,
          - final_noise_count, and
          - target_numbers_scaled (the scaled target corrections per error type).

        The target numbers are scaled (with int conversion) and then used to adjust the counts.
        """
        # Scale target numbers and convert to int.
        target_numbers_scaled = {
            key: int(A * value) for key, value in target_numbers.items()
        }

        # Compute corrections (i.e. the difference between actual error counts and scaled target counts).
        # Correct counts: +2 for "multiple_entities", +1 for "wrong_label", "wrong_overlap", "missing"
        added_correct = (
            (
                error_analysis["multiple_entities"]
                - target_numbers_scaled["multiple_entities"]
            )
            * 2
            + (error_analysis["wrong_label"] - target_numbers_scaled["wrong_label"])
            + (error_analysis["wrong_overlap"] - target_numbers_scaled["wrong_overlap"])
            + (error_analysis["missing"] - target_numbers_scaled["missing"])
        )

        # Noise reduction: remove -1 for each correction for "multiple_entities", "wrong_label", "wrong_overlap", "invalid_tag"
        reduced_noise = (
            (
                error_analysis["multiple_entities"]
                - target_numbers_scaled["multiple_entities"]
            )
            + (error_analysis["wrong_label"] - target_numbers_scaled["wrong_label"])
            + (error_analysis["wrong_overlap"] - target_numbers_scaled["wrong_overlap"])
            + (error_analysis["invalid_tag"] - target_numbers_scaled["invalid_tag"])
        )

        final_correct = initial_correct_counts + added_correct
        final_noise_count = initial_noise_counts - reduced_noise

        # Compute precision, recall, and F1 score.
        denom = final_correct + final_noise_count
        precision = final_correct / denom if denom > 0 else 0
        recall = final_correct / n_true_spans if n_true_spans > 0 else 0
        f1 = (
            (2 * precision * recall / (precision + recall))
            if (precision + recall) > 0
            else 0
        )

        return 1 - f1, final_correct, final_noise_count, target_numbers_scaled

    # f(A) is the difference between computed final noise and wanted_noise.
    def f(A: float) -> float:
        noise, _, _, _ = final_metrics(A)
        return noise - wanted_noise

    # Initialize search bounds for A.
    A_low = 0.0
    A_high = 1.0
    # Expand A_high until f(A_low) and f(A_high) have opposite signs.
    while f(A_low) * f(A_high) > 0 and A_high < 1e6:
        A_high *= 2

    if A_high >= 1e6:
        raise ValueError("Could not find suitable bounds for A.")

    # Bisection method parameters.
    tol = 1e-4
    max_iter = 100
    A_mid = (A_low + A_high) / 2.0

    for i in range(max_iter):
        noise_mid, correct_mid, noise_count_mid, scaled_targets = final_metrics(A_mid)
        error_val = noise_mid - wanted_noise

        if debug:
            print(
                f"Iteration {i}: A = {A_mid:.4f}, final_noise = {noise_mid:.4f}, error = {error_val:.4f}"
            )

        if abs(error_val) < tol:
            break

        if f(A_low) * error_val < 0:
            A_high = A_mid
        else:
            A_low = A_mid

        A_mid = (A_low + A_high) / 2.0

    final_noise, final_correct, final_noise_count, final_target_numbers_scaled = (
        final_metrics(A_mid)
    )
    return (
        A_mid,
        final_noise,
        final_correct,
        final_noise_count,
        final_target_numbers_scaled,
    )


def reduce_noise_distribution(
    full_results,
    wanted_noise,
    reduction,
    total_n_true,
    total_noise_types,
    target_numbers,
    debug=False,
):
    """
    Similar to "reduce_noise" function but such that the remaining errors follow the given target numbers.

    Parameters:
      - full_results: list of dicts, each corresponding to an abstract. Each dict should contain:
            "pmid": str,
            "noise": float,        # noise ratio for the abstract (computed as 1 - f1)
            "noise_count": int,     # number of noisy spans in the abstract
            "correct_count": int,   # number of correct spans in the abstract
            "n_true": int,          # total true spans in the abstract
            "pred_spans": list,     # list of predicted spans
            "true_spans": list,     # list of true spans
            "results": list         # list of dicts with keys: start, end, label, tag, text, correct, error_type
      - wanted_noise: float, overall noise ratio desired for the dataset.
            For example, if you want a final noise of 0.10 (i.e. f1 = 0.90).
      - reduction: float, fraction by which to reduce noise per abstract.
            For instance, reduction=0.5 will remove about half of the noisy spans (subject to per-error-type constraints).
      - total_n_true: int, total number of true spans in the whole dataset.
      - total_noise_types: dict, counts of errors per type for the whole dataset.
            Example: {"invalid_tag": 29839, "wrong_label": 12475, "missing": 17031, "wrong_overlap": 16693, "multiple_entities": 8658}
      - target_numbers: dict, desired final count for each error type.
            Example: {"wrong_label": 7323, "wrong_overlap": 10984, "multiple_entities": 4882, "invalid_tag": 14646, "missing": 10984}
      - debug: bool, if True prints debugging information.

    Returns:
      - new_full_results: list of dicts similar to full_results but with noise reduced per abstract.
    """
    # Step 1: Set up the global removal budget.
    # For each error type, compute how many instances should be removed
    # globally as: total available - desired final count.
    errors_to_remove_global = {}
    for err_type, total in total_noise_types.items():
        if err_type not in target_numbers:
            # If no target is provided, we do not remove any instances for this error type.
            errors_to_remove_global[err_type] = 0
        else:
            desired_final = target_numbers[err_type]
            if desired_final > total:
                raise AssertionError(
                    f"Target number for error type '{err_type}' is greater than the total available. "
                    f"Target: {desired_final}, total: {total}."
                )
            errors_to_remove_global[err_type] = total - desired_final

    if debug:
        print("Global errors to remove (per type):", errors_to_remove_global)

    # Step 2: Compute overall counts and metrics.
    current_noise_count = sum(item["noise_count"] for item in full_results)
    current_correct_count = sum(item["correct_count"] for item in full_results)

    precision = (
        current_correct_count / (current_correct_count + current_noise_count)
        if (current_correct_count + current_noise_count) > 0
        else 0
    )
    recall = current_correct_count / total_n_true if total_n_true > 0 else 0
    f1 = (
        (2 * precision * recall / (precision + recall))
        if (precision + recall) > 0
        else 0
    )
    current_noise = 1 - f1

    print("Initial correct counts :", current_correct_count)
    print("Initial noise counts :", current_noise_count)
    print("Initial global noise :", current_noise)

    # Step 3: Process abstracts in random order.
    abstracts = full_results.copy()
    random.shuffle(abstracts)
    new_full_results = []
    idx = 0

    while current_noise > wanted_noise and idx < len(abstracts):
        abstract = abstracts[idx]

        # Calculate error type counts in this abstract.
        abs_noise_types = {}
        for span in abstract["results"]:
            if not span["correct"]:
                err = span["error_type"]
                abs_noise_types[err] = abs_noise_types.get(err, 0) + 1

        # Build a per-abstract removal plan.
        # For each error type in this abstract, aim to remove a fraction (given by reduction)
        # but do not remove more than:
        #   - the number available (leaving at least one if count > 1), and
        #   - the remaining global removal budget.
        errors_to_remove_abstract = {}
        # for err, count in abs_noise_types.items():
        #     if err in errors_to_remove_global:
        #         candidate = int(round(reduction * count))
        #         candidate = min(candidate, errors_to_remove_global[err])
        #         if count > 1:
        #             candidate = min(candidate, count - 1)
        #         else:
        #             candidate = 0
        #         errors_to_remove_abstract[err] = candidate
        #     else:
        #         errors_to_remove_abstract[err] = 0
        for err, count in abs_noise_types.items():
            if err in errors_to_remove_global:
                candidate = int(round(reduction * count))
                candidate = min(candidate, errors_to_remove_global[err])
                errors_to_remove_abstract[err] = candidate
            else:
                errors_to_remove_abstract[err] = 0

        if debug:
            print("Processing abstract pmid:", abstract["pmid"])
            print("  Abstract error counts:", abs_noise_types)
            print("  Errors to remove in abstract:", errors_to_remove_abstract)

        # Reduce noise in this abstract based on the per-error removal plan.
        new_spans = reduce_noise_per_abstract_distribution(
            correct_count=abstract["correct_count"],
            noise_count=abstract["noise_count"],
            pred_spans=abstract["pred_spans"],
            true_spans=abstract["true_spans"],
            reduction=reduction,
            results=abstract["results"],
            noise_types=abs_noise_types,
            errors_to_remove=errors_to_remove_abstract,
            debug=False,
        )
        # Compute the new metrics for this abstract.
        (
            new_f1,
            new_precision,
            new_recall,
            new_correct_count,
            new_noise_count,
            new_results,
            new_abs_noise_types,
        ) = compute_noise(new_spans, abstract["true_spans"])

        # Update the global removal budget based on what was removed.
        for err in errors_to_remove_global.keys():
            original = abs_noise_types.get(err, 0)
            new_count = new_abs_noise_types.get(err, 0)
            removed = original - new_count
            errors_to_remove_global[err] = max(
                errors_to_remove_global[err] - removed, 0
            )

        # Update global counts.
        reduction_in_abstract_noise = abstract["noise_count"] - new_noise_count
        reduction_in_abstract_correct = abstract["correct_count"] - new_correct_count
        current_noise_count -= reduction_in_abstract_noise
        current_correct_count -= reduction_in_abstract_correct

        # Recompute global metrics.
        precision = (
            current_correct_count / (current_correct_count + current_noise_count)
            if (current_correct_count + current_noise_count) > 0
            else 0
        )
        recall = current_correct_count / total_n_true if total_n_true > 0 else 0
        f1 = (
            (2 * precision * recall / (precision + recall))
            if (precision + recall) > 0
            else 0
        )
        current_noise = 1 - f1

        updated_abstract = {
            "pmid": abstract["pmid"],
            "noise": 1 - new_f1,
            "f1": new_f1,
            "precision": new_precision,
            "recall": new_recall,
            "correct_count": new_correct_count,
            "noise_count": new_noise_count,
            "pred_spans": new_spans,
            "true_spans": abstract["true_spans"],
            "results": new_results,
            "noise_types": new_abs_noise_types,
        }
        new_full_results.append(updated_abstract)
        if debug:
            print(
                f"  Updated abstract noise count: {new_noise_count} (Former {abstract['noise_count']})"
            )
            print(
                f"  Updated abstract correct count: {new_correct_count} (Former {abstract['correct_count']})"
            )
            print("  Global remaining removal budget:", errors_to_remove_global)
            print("  Global current noise:", current_noise)
        idx += 1

    print("Final correct count :", current_correct_count)
    print("Final noise count :", current_noise_count)
    print("Final global noise:", current_noise)
    print("Number of abstracts with noise reduced:", idx)

    # Append remaining abstracts without further reduction.
    while idx < len(abstracts):
        new_full_results.append(abstracts[idx])
        idx += 1

    return new_full_results
