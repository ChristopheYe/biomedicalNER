import numpy as np
import ujson
from itertools import chain
import pandas as pd
from datasets import load_dataset
from collections import Counter, defaultdict
import warnings

warnings.filterwarnings("ignore")

subsets_data = {
    20: [
        "27289224",
        "27333086",
        "27348675",
        "27374919",
        "27413283",
        "27436508",
        "27457924",
        "27617056",
        "27677676",
        "27778115",
        "27787497",
        "27802069",
        "27856909",
        "27873364",
        "27942475",
        "28249880",
        "28263315",
        "28341154",
        "28375653",
        "28392546",
    ],
    40: [
        "27260358",
        "27282236",
        "27289224",
        "27333086",
        "27348675",
        "27374109",
        "27374919",
        "27391139",
        "27393884",
        "27413283",
        "27423415",
        "27436508",
        "27570556",
        "27677676",
        "27704621",
        "27778115",
        "27787497",
        "27802069",
        "27851791",
        "27856909",
        "27873364",
        "27899295",
        "27903877",
        "27942475",
        "27975319",
        "28017917",
        "28151970",
        "28250022",
        "28263315",
        "28265773",
        "28299685",
        "28330740",
        "28341154",
        "28343654",
        "28345195",
        "28375653",
        "28388318",
        "28392546",
        "28468745",
        "28470421",
    ],
    60: [
        "26864880",
        "27256126",
        "27282236",
        "27289224",
        "27333086",
        "27348675",
        "27374109",
        "27374919",
        "27393884",
        "27411238",
        "27413283",
        "27423415",
        "27436508",
        "27457924",
        "27570556",
        "27615860",
        "27617056",
        "27690192",
        "27704621",
        "27778115",
        "27787497",
        "27802069",
        "27802781",
        "27830257",
        "27851791",
        "27856909",
        "27873364",
        "27899295",
        "27903877",
        "27935268",
        "27941707",
        "27942475",
        "27975319",
        "28017917",
        "28123271",
        "28133624",
        "28151970",
        "28165155",
        "28245088",
        "28248975",
        "28250022",
        "28263315",
        "28265773",
        "28330740",
        "28339013",
        "28341154",
        "28343654",
        "28375653",
        "28377331",
        "28388318",
        "28391771",
        "28392546",
        "28418171",
        "28435583",
        "28458290",
        "28462652",
        "28468745",
        "28497407",
        "28518216",
        "28544860",
    ],
    100: [
        "26864880",
        "27256126",
        "27256373",
        "27260358",
        "27282236",
        "27289224",
        "27312379",
        "27333086",
        "27339000",
        "27348675",
        "27374109",
        "27374919",
        "27391139",
        "27393884",
        "27411238",
        "27413283",
        "27423415",
        "27431445",
        "27436508",
        "27457924",
        "27464911",
        "27481810",
        "27522061",
        "27555654",
        "27570556",
        "27580729",
        "27615860",
        "27617056",
        "27677676",
        "27683502",
        "27690192",
        "27704621",
        "27721673",
        "27744673",
        "27749441",
        "27778115",
        "27787497",
        "27802069",
        "27802781",
        "27811200",
        "27830257",
        "27832736",
        "27851791",
        "27856909",
        "27873364",
        "27899295",
        "27903877",
        "27935268",
        "27941707",
        "27942475",
        "27975319",
        "28004480",
        "28012388",
        "28017917",
        "28045933",
        "28050352",
        "28088240",
        "28105538",
        "28108814",
        "28123271",
        "28125006",
        "28128305",
        "28133624",
        "28143537",
        "28151970",
        "28190724",
        "28245088",
        "28248975",
        "28249880",
        "28250022",
        "28263315",
        "28265773",
        "28280246",
        "28284588",
        "28299685",
        "28327645",
        "28330740",
        "28339013",
        "28341154",
        "28343654",
        "28345195",
        "28362410",
        "28375653",
        "28377331",
        "28388318",
        "28391771",
        "28392546",
        "28418171",
        "28421317",
        "28435583",
        "28442480",
        "28456743",
        "28456744",
        "28458290",
        "28462652",
        "28468745",
        "28470421",
        "28497407",
        "28518216",
        "28544860",
    ],
    200: [
        "26864880",
        "27242101",
        "27250037",
        "27253877",
        "27256126",
        "27256373",
        "27260358",
        "27265179",
        "27282236",
        "27289224",
        "27293123",
        "27293431",
        "27299182",
        "27330705",
        "27333086",
        "27337704",
        "27339000",
        "27348675",
        "27349321",
        "27374109",
        "27374919",
        "27381890",
        "27385839",
        "27391139",
        "27393884",
        "27397797",
        "27399843",
        "27411238",
        "27413283",
        "27423415",
        "27424514",
        "27426713",
        "27426888",
        "27431445",
        "27436508",
        "27440478",
        "27453359",
        "27457924",
        "27464911",
        "27478557",
        "27481810",
        "27512256",
        "27512378",
        "27522061",
        "27555654",
        "27561345",
        "27570556",
        "27580729",
        "27585205",
        "27590051",
        "27606118",
        "27615860",
        "27617056",
        "27622096",
        "27630041",
        "27653361",
        "27677676",
        "27683502",
        "27690192",
        "27694049",
        "27704621",
        "27721673",
        "27730636",
        "27742821",
        "27744673",
        "27749441",
        "27774519",
        "27777712",
        "27778115",
        "27784172",
        "27787497",
        "27787956",
        "27788427",
        "27794223",
        "27794611",
        "27802069",
        "27802781",
        "27811200",
        "27811711",
        "27813599",
        "27818989",
        "27822302",
        "27830257",
        "27832736",
        "27848155",
        "27851791",
        "27856909",
        "27857099",
        "27859172",
        "27873364",
        "27873506",
        "27875815",
        "27891554",
        "27899295",
        "27903877",
        "27912013",
        "27931138",
        "27935268",
        "27941707",
        "27942475",
        "27975319",
        "27977490",
        "27994743",
        "28002976",
        "28004480",
        "28012388",
        "28017917",
        "28038326",
        "28039389",
        "28045933",
        "28050352",
        "28055578",
        "28056457",
        "28065558",
        "28070315",
        "28082632",
        "28087071",
        "28088240",
        "28105538",
        "28108814",
        "28117729",
        "28117911",
        "28118389",
        "28123271",
        "28125006",
        "28126550",
        "28128305",
        "28133624",
        "28143537",
        "28151970",
        "28159617",
        "28163489",
        "28165155",
        "28168230",
        "28177801",
        "28181519",
        "28190724",
        "28236014",
        "28238845",
        "28244050",
        "28245088",
        "28246046",
        "28248975",
        "28249880",
        "28250022",
        "28261010",
        "28263315",
        "28265773",
        "28266760",
        "28266784",
        "28274448",
        "28280246",
        "28284588",
        "28299685",
        "28306707",
        "28316359",
        "28327645",
        "28330740",
        "28337431",
        "28339013",
        "28341154",
        "28341723",
        "28343654",
        "28345195",
        "28359781",
        "28362410",
        "28366770",
        "28375653",
        "28377331",
        "28388318",
        "28390302",
        "28391771",
        "28392546",
        "28409856",
        "28418171",
        "28421317",
        "28423778",
        "28435583",
        "28436738",
        "28442480",
        "28454863",
        "28456743",
        "28456744",
        "28456850",
        "28458290",
        "28458859",
        "28462652",
        "28468745",
        "28470421",
        "28481498",
        "28490253",
        "28497407",
        "28497418",
        "28499343",
        "28500879",
        "28518216",
        "28533814",
        "28535279",
        "28539075",
        "28544860",
    ],
    263: [
        "26864880",
        "27242101",
        "27250037",
        "27253877",
        "27256126",
        "27256373",
        "27265179",
        "27268023",
        "27282236",
        "27288403",
        "27289224",
        "27293123",
        "27293431",
        "27299182",
        "27312379",
        "27330705",
        "27333086",
        "27337704",
        "27339000",
        "27344476",
        "27348675",
        "27349321",
        "27359321",
        "27374109",
        "27374919",
        "27378064",
        "27381890",
        "27385839",
        "27391139",
        "27393884",
        "27397797",
        "27399843",
        "27411238",
        "27413283",
        "27423415",
        "27424514",
        "27426713",
        "27426888",
        "27431445",
        "27436508",
        "27440478",
        "27445402",
        "27451934",
        "27453359",
        "27457924",
        "27464911",
        "27465497",
        "27478557",
        "27481810",
        "27505348",
        "27512256",
        "27512378",
        "27514440",
        "27521980",
        "27522061",
        "27555654",
        "27561345",
        "27570556",
        "27578894",
        "27580729",
        "27585205",
        "27590051",
        "27606118",
        "27615860",
        "27617056",
        "27617383",
        "27622096",
        "27630041",
        "27650514",
        "27653361",
        "27661391",
        "27677398",
        "27677676",
        "27683502",
        "27690192",
        "27694049",
        "27704621",
        "27721673",
        "27730636",
        "27742821",
        "27744673",
        "27749441",
        "27755647",
        "27774519",
        "27776603",
        "27777712",
        "27778115",
        "27784172",
        "27787497",
        "27787956",
        "27788427",
        "27794223",
        "27794611",
        "27799038",
        "27802069",
        "27802781",
        "27811200",
        "27811711",
        "27813599",
        "27818989",
        "27822302",
        "27827387",
        "27830257",
        "27832736",
        "27848155",
        "27851729",
        "27851791",
        "27856909",
        "27857099",
        "27859172",
        "27873364",
        "27873506",
        "27875815",
        "27891554",
        "27898317",
        "27899295",
        "27900888",
        "27903877",
        "27907227",
        "27908578",
        "27912013",
        "27913270",
        "27920636",
        "27924533",
        "27931138",
        "27935268",
        "27941707",
        "27942475",
        "27975319",
        "27977490",
        "27984098",
        "27994743",
        "28002976",
        "28003799",
        "28004480",
        "28012388",
        "28017917",
        "28038326",
        "28039389",
        "28041848",
        "28045933",
        "28050352",
        "28051899",
        "28055578",
        "28056457",
        "28065558",
        "28070315",
        "28082632",
        "28087071",
        "28088240",
        "28089332",
        "28108814",
        "28110451",
        "28117729",
        "28117911",
        "28118389",
        "28123271",
        "28125006",
        "28126110",
        "28126550",
        "28128305",
        "28133624",
        "28135680",
        "28138777",
        "28143537",
        "28151970",
        "28159617",
        "28161688",
        "28163489",
        "28165155",
        "28166581",
        "28167269",
        "28168230",
        "28177801",
        "28181519",
        "28190724",
        "28202626",
        "28214821",
        "28235434",
        "28236014",
        "28238845",
        "28244050",
        "28245088",
        "28246046",
        "28248564",
        "28248975",
        "28249880",
        "28250022",
        "28261010",
        "28263315",
        "28265773",
        "28266760",
        "28266784",
        "28267102",
        "28274448",
        "28280246",
        "28282937",
        "28284588",
        "28287597",
        "28299685",
        "28306707",
        "28316359",
        "28318461",
        "28319277",
        "28327645",
        "28330740",
        "28337431",
        "28339013",
        "28340260",
        "28341154",
        "28341623",
        "28341723",
        "28343654",
        "28345195",
        "28353232",
        "28359781",
        "28362410",
        "28362864",
        "28366770",
        "28374841",
        "28375653",
        "28377331",
        "28388318",
        "28389961",
        "28390302",
        "28391771",
        "28392546",
        "28409856",
        "28418171",
        "28421317",
        "28423778",
        "28435583",
        "28436738",
        "28442480",
        "28452185",
        "28454863",
        "28456743",
        "28456744",
        "28456850",
        "28458290",
        "28458859",
        "28462652",
        "28468745",
        "28469905",
        "28470421",
        "28470550",
        "28481498",
        "28487909",
        "28490253",
        "28493821",
        "28497407",
        "28497418",
        "28498873",
        "28499343",
        "28500879",
        "28508736",
        "28512111",
        "28518216",
        "28533814",
        "28535279",
        "28539075",
        "28544860",
        "28547383",
    ],
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

tag2label_bc5cdr_chemical = {"Chemical": 0}
extraction_prompts_bc5cdr_chemical = {
    "Chemical": "Return the text that is related to chemical. Example usage: in 'Suxamethonium infusion rate and observed fasciculations.', return 'Suxamethonium'."
}

tag2label_bc5cdr_disease = {"Disease": 0}
extraction_prompts_bc5cdr_disease = {
    "Disease": "Return the text that is related to disease. Example usage: in 'Suxamethonium infusion rate and observed fasciculations.', return 'fasciculations'."
}

tag2label_bc5cdr = {
    "Chemical": 0,
    "Disease": 1,
}
extraction_prompts_bc5cdr = {
    "Chemical": "Return the text that is related to chemical. Example usage: in 'Suxamethonium infusion rate and observed fasciculations.', return 'Suxamethonium'.",
    "Disease": "Return the text that is related to disease. Example usage: in 'Suxamethonium infusion rate and observed fasciculations.', return 'fasciculations'.",
}

tag2label_pico = {
    "participant": 0,
    "intervention": 1,
    "outcome": 2,
}

extraction_prompts_pico = {
    "Participant": (
        "Return the text that refers to the patient/participant/population involved in the clinical study. "
        "Example: in 'Anti-emetic efficacy of prophylactic granisetron compared with perphenazine for the prevention of post-operative vomiting in children.', return 'children'."
    ),
    "Intervention": (
        "Return the text describing the intervention(s) administered in the clinical study. "
        "Example: in 'Anti-emetic efficacy of prophylactic granisetron compared with perphenazine for the prevention of post-operative vomiting in children.', return 'granisetron' and 'perphenazine'."
    ),
    "Outcome": (
        "Return the text describing the outcome(s) measured or observed in the study. "
        "Example: in 'Anti-emetic efficacy of prophylactic granisetron compared with perphenazine for the prevention of post-operative vomiting in children.', return 'Anti-emetic efficacy' and 'prevention'."
    ),
}

tag2label_ncbi = {
    "Disease": 0,
}

extraction_prompts_ncbi = {
    "Disease": "Return the text that is related to disease. Example usage: in 'Suxamethonium infusion rate and observed fasciculations.', return 'fasciculations'.",
}

# Used for loading already trained model
model_params = {
    "gpt4o_noise=0.1": {  # BiolinkBERT trained on GPT4o predictions with noise reduced to 10%
        "model_subname": "gpt-4o-2024-11-20_dynamic=True_k=10_v11_noise=0.1_BioLinkBERT-base",
        "checkpoint": "checkpoint-2494",
    },
    "gpt4o_noise=0.2": {
        "model_subname": "gpt-4o-2024-11-20_dynamic=True_k=10_v11_noise=0.2_BioLinkBERT-base",
        "checkpoint": "checkpoint-2322",
    },
    "gpt4o_noise=0.3": {
        "model_subname": "gpt-4o-2024-11-20_dynamic=True_k=10_v11_noise=0.3_BioLinkBERT-base",
        "checkpoint": "checkpoint-2322",
    },
    "gpt4o_noise=0.4": {
        "model_subname": "gpt-4o-2024-11-20_dynamic=True_k=10_v11_noise=0.4_BioLinkBERT-base",
        "checkpoint": "checkpoint-2408",
    },
    "gpt4o_noise=0.5": {
        "model_subname": "gpt-4o-2024-11-20_dynamic=True_k=10_v11_noise=0.5_BioLinkBERT-base",
        "checkpoint": "checkpoint-2580",
    },
    "gpt4o_noise=0.6": {
        "model_subname": "gpt-4o-2024-11-20_dynamic=True_k=10_v11_noise=0.6_BioLinkBERT-base",
        "checkpoint": "checkpoint-2064",
    },
    "gpt4o_noise=0.694": {
        "model_subname": "gpt-4o-2024-11-20_dynamic=True_k=10_v11_noise=False_BioLinkBERT-base",
        "checkpoint": "checkpoint-2322",
    },
    "phi4_noise=0.1": {  # BiolinkBERT trained on Phi4 predictions with noise reduced to 10%
        "model_subname": "phi-4_v15_noise=0.1_BioLinkBERT-base",
        "checkpoint": "checkpoint-2408",
    },
    "phi4_noise=0.2": {
        "model_subname": "phi-4_v15_noise=0.2_BioLinkBERT-base",
        "checkpoint": "checkpoint-2494",
    },
    "phi4_noise=0.3": {
        "model_subname": "phi-4_v15_noise=0.3_BioLinkBERT-base",
        "checkpoint": "checkpoint-258",
    },
    "phi4_noise=0.4": {
        "model_subname": "phi-4_v15_noise=0.4_BioLinkBERT-base",
        "checkpoint": "checkpoint-516",
    },
    "phi4_noise=0.5": {
        "model_subname": "phi-4_v15_noise=0.5_BioLinkBERT-base",
        "checkpoint": "checkpoint-774",
    },
    "phi4_noise=0.6": {
        "model_subname": "phi-4_v15_noise=0.6_BioLinkBERT-base",
        "checkpoint": "checkpoint-258",
    },
    "phi4_noise=0.7": {
        "model_subname": "phi-4_v15_noise=0.7_BioLinkBERT-base",
        "checkpoint": "checkpoint-1634",
    },
    "gemma3_noise=0.1": {  # BiolinkBERT trained on Gemma3 predictions with noise reduced to 10%
        "model_subname": "gemma-3-12b-it_v2_noise=0.1_BioLinkBERT-base_v1",
        "checkpoint": "checkpoint-2580",
    },
    "gemma3_noise=0.2": {
        "model_subname": "gemma-3-12b-it_v2_noise=0.2_BioLinkBERT-base_v1",
        "checkpoint": "checkpoint-1118",
    },
    "gemma3_noise=0.3": {
        "model_subname": "gemma-3-12b-it_v2_noise=0.3_BioLinkBERT-base_v1",
        "checkpoint": "checkpoint-1204",
    },
    "gemma3_noise=0.4": {
        "model_subname": "gemma-3-12b-it_v2_noise=0.4_BioLinkBERT-base_v1",
        "checkpoint": "checkpoint-2150",
    },
    "gemma3_noise=0.5": {
        "model_subname": "gemma-3-12b-it_v2_noise=0.5_BioLinkBERT-base_v1",
        "checkpoint": "checkpoint-1978",
    },
    "gemma3_noise=0.6": {
        "model_subname": "gemma-3-12b-it_v2_noise=0.6_BioLinkBERT-base_v1",
        "checkpoint": "checkpoint-1892",
    },
    "gemma3_noise=0.7": {
        "model_subname": "gemma-3-12b-it_v2_noise=0.7_BioLinkBERT-base_v1",
        "checkpoint": "checkpoint-1032",
    },
    "mistral3_noise=0.1": {  # BiolinkBERT trained on Mistral predictions with noise reduced to 10%
        "model_subname": "Mistral-Small-3.1-24B-Instruct-2503_dynamic=True_k=5_v3_noise=0.1_BioLinkBERT-base_v1",
        "checkpoint": "checkpoint-4959",
    },
    "mistral3_noise=0.2": {
        "model_subname": "Mistral-Small-3.1-24B-Instruct-2503_dynamic=True_k=5_v3_noise=0.2_BioLinkBERT-base_v1",
        "checkpoint": "checkpoint-684",
    },
    "mistral3_noise=0.3": {
        "model_subname": "Mistral-Small-3.1-24B-Instruct-2503_dynamic=True_k=5_v3_noise=0.3_BioLinkBERT-base_v1",
        "checkpoint": "checkpoint-5130",
    },
    "mistral3_noise=0.4": {
        "model_subname": "Mistral-Small-3.1-24B-Instruct-2503_dynamic=True_k=5_v3_noise=0.4_BioLinkBERT-base_v1",
        "checkpoint": "checkpoint-513",
    },
    "mistral3_noise=0.5": {
        "model_subname": "Mistral-Small-3.1-24B-Instruct-2503_dynamic=True_k=5_v3_noise=0.5_BioLinkBERT-base_v1",
        "checkpoint": "checkpoint-684",
    },
    "mistral3_noise=0.6": {
        "model_subname": "Mistral-Small-3.1-24B-Instruct-2503_dynamic=True_k=5_v3_noise=0.6_BioLinkBERT-base_v1",
        "checkpoint": "checkpoint-4788",
    },
    "mistral3_noise=0.7": {
        "model_subname": "Mistral-Small-3.1-24B-Instruct-2503_dynamic=True_k=5_v3_noise=0.7_BioLinkBERT-base_v1",
        "checkpoint": "checkpoint-5130",
    },
}


def load_bigbio_dataset(dataset_name):
    """
    Load BigBio dataset and include abbreviations if specified.

    Params
    ------
    - dataset_name : str
        Name of the dataset to load
    """
    # Load the dataset
    if dataset_name in {"medmentions_st21pv", "medmentions_full"}:
        dataset = load_dataset(
            f"bigbio/medmentions",
            name=f"{dataset_name}_bigbio_kb",
            trust_remote_code=True,
        )
    else:
        dataset = load_dataset(
            f"bigbio/{dataset_name}",
            name=f"{dataset_name}_bigbio_kb",
            trust_remote_code=True,
        )

    return dataset


def dataset_to_df(
    dataset,
    splits_to_include: list = None,
    entity_remapping_dict: dict = None,
    cuis_to_exclude: list = None,
    val_split_ids: list = None,
    test_split_ids: list = None,
    dataset_name: str = None,
    # abbreviations_dict: dict = None,
):
    """
    Convert BigBio dataset to pandas DataFrame

    Params:
    ------------------
        dataset: BigBio Dataset
            Dataset to load from BigBio

        splits_to_include: list of str
            List of splits to include in mo
    """
    columns = [
        # 'context', # string
        "document_id",  # string
        "mention_id",  # string
        "text",  # string
        "type",  # list
        "offsets",  # list of lists
        # "db_name",
        "db_ids",  # list
        "split",  # string
        # "abbreviation_resolved", # bool
        "deabbreviated_text",  # string
    ]
    all_lines = []

    if splits_to_include is None:
        splits_to_include = dataset.keys()

    for split in splits_to_include:
        if split not in dataset.keys():
            warnings.warn(f"Split '{split}' not in dataset.  Omitting.")
        for doc in dataset[split]:
            pmid = doc["document_id"]
            for e in doc["entities"]:
                if len(e["normalized"]) == 0 and dataset_name not in [
                    "ebm_pico",
                    "pico_extraction",
                ]:
                    continue
                text = " ".join(e["text"])
                # abbreviation_resolved = False
                offsets = ";".join(
                    [",".join([str(y) for y in x]) for x in e["offsets"]]
                )
                # db_name = e["normalized"][0]["db_name"]
                db_ids = [
                    x["db_name"] + ":" + x["db_id"].strip() for x in e["normalized"]
                ]

                # Get the abbreviation if it exists, else set to None or an empty string
                deabbreviated_text = e.get("deabbreviated_text", None)

                # Remap entity IDs when identifier has changed in database
                if entity_remapping_dict is not None:
                    db_ids = [
                        entity_remapping_dict[x] if x in entity_remapping_dict else x
                        for x in db_ids
                    ]

                # Remove any IDs not included in database
                # Remove mentions with no entity link in database
                if cuis_to_exclude is not None:
                    new_db_ids = [x for x in db_ids if x not in cuis_to_exclude]
                else:
                    new_db_ids = db_ids
                if len(new_db_ids) == 0 and dataset_name not in [
                    "ebm_pico",
                    "pico_extraction",
                ]:
                    continue

                # Add mention + metadata to list of mentions
                all_lines.append(
                    [
                        pmid,
                        e["id"],
                        text,
                        e["type"],
                        # e['offsets'],
                        offsets,
                        # db_name,
                        new_db_ids,
                        split,
                        # abbreviation_resolved,
                        deabbreviated_text,
                    ]
                )

    df = pd.DataFrame(all_lines, columns=columns)

    deduplicated = (
        df.groupby(["document_id", "offsets"])
        .agg(
            {
                "text": "first",
                "type": lambda x: list(set([a for a in x])),
                "db_ids": lambda db_ids: list(set([y for x in db_ids for y in x])),
                "split": "first",
                "deabbreviated_text": "first",
            }
        )
        .reset_index()
    )

    deduplicated["offsets"] = deduplicated["offsets"].map(
        lambda x: [[int(z) for z in y.split(",")] for y in x.split(";")]
    )

    # Order mentions in consistent way (i.e. )
    deduplicated["first_offset"] = deduplicated["offsets"].map(lambda x: x[0][0])
    deduplicated["last_offset"] = deduplicated["offsets"].map(lambda x: x[-1][-1])
    deduplicated = deduplicated.sort_values(
        by=[
            "document_id",
            "first_offset",
            "last_offset",
        ]
    )
    deduplicated = deduplicated.drop(["first_offset", "last_offset"], axis=1)

    # Make a unique mention ID for each mention
    mention_counts = defaultdict(int)
    deduplicated["mention_id"] = deduplicated["document_id"].map(
        lambda x: make_mention_id(x, running_mention_count=mention_counts)
    )

    # Split off validation set if not given
    if val_split_ids is not None:
        # print(type(val_split_ids[0]), type(deduplicated["document_id"][0]))
        deduplicated.loc[deduplicated["document_id"].isin(val_split_ids), "split"] = (
            "validation"
        )
    if test_split_ids is not None:
        deduplicated.loc[deduplicated["document_id"].isin(test_split_ids), "split"] = (
            "test"
        )
    return deduplicated


def make_mention_id(document_id, running_mention_count):
    """
    Make a unique ID for each mention
    """
    running_mention_count[document_id] += 1
    return f"{document_id}.{running_mention_count[document_id]}"


def dataset_to_documents(dataset):
    """
    Return dictionary of documents in BigBio dataset
    """
    docs = {}

    for split in dataset.keys():
        for doc in dataset[split]:
            doc_id = pmid = doc["document_id"]
            doc_text = "\n".join([" ".join(x["text"]) for x in doc["passages"]])
            docs[doc_id] = doc_text
    return docs


# def process_dataset(docs, df):  # NO LONGER USED / REPLACED BY replace_text_with_tags
#     """
#     Function to create a list of data where each data is in this format :
#     {'text':text, (str)
#         'spans':spans, (list of dict)
#         'pmid':pmid, (int)
#         'split':split (str)
#     }
#     where spans will have form:
#     {'start':int,
#         'end':int,
#         'label':int,
#         'tag':str,
#         'text':str
#     }
#     -------------
#     docs : dict {pmid : abstract}
#     df : pd.DataFrame
#     tag2label : dict {tag : label}
#     """
#     processed_data = []
#     unique_values = list(set(chain.from_iterable(df["type"])))
#     tag2label = {tag: label for label, tag in enumerate(unique_values)}
#     for pmid in docs:
#         doc = {}
#         doc["text"] = docs[pmid]
#         doc["spans"] = []
#         sub_df = df[df["document_id"] == pmid]
#         for idx, row in sub_df.iterrows():
#             start = row["offsets"][0][0]  # start on the mention
#             end = row["offsets"][-1][-1]  # end of the mention
#             label = tag2label[row["type"][0]]  # label
#             tag = row["type"][0]  # tag
#             text = row["text"]  # text
#             span = {
#                 "start": start,
#                 "end": end,
#                 "label": label,
#                 "tag": tag,
#                 "text": text,
#             }
#             doc["spans"].append(span)
#         doc["pmid"] = pmid
#         doc["split"] = row["split"]
#         processed_data.append(doc)
#     return processed_data


def process_dataset(docs, df, tag2label):
    """
    Function which modifies the abstract to include the entity:
    Input : DCTN4 as a modifier of chronic Pseudomonas aeruginosa infection
    Output : @@DCTN4##T103@@ as a modifier of @@chronic Pseudomonas aeruginosa infection##T038@@
    -------------
    docs : dict {pmid : abstract}
    df : pd.DataFrame
    """
    processed_data = {}
    for pmid in docs:
        # Get the specific abstract and filter dataframe for the given document ID
        new_df = df[df["document_id"] == pmid]
        abstract = docs[pmid]

        # Sort dataframe by the start offset to handle replacements in order
        new_df["start"] = new_df["offsets"].apply(lambda x: x[0][0])
        df_sorted = new_df.sort_values(
            by="start", ascending=False
        )  # Replace from the end to avoid offset shifts

        # Perform replacements based on offsets
        spans = []
        for _, row in df_sorted.iterrows():
            start, end = row["offsets"][0]
            text = row["text"]
            label = tag2label[row["type"][0]]  # label
            entity_type = str(row["type"][0])  # type
            # print("entity_type :", entity_type)
            tagged_text = f"@@{text}##{entity_type}@@"
            tag = row["type"][0]
            span = {
                "start": start,
                "end": end,
                "label": label,
                "tag": tag,
                "text": text,
            }
            spans.append(span)
            # Replace the text at the specific offset
            abstract = abstract[:start] + tagged_text + abstract[end:]

        processed_data[pmid] = {
            "input": docs[pmid],
            "output": abstract,
            "spans": spans[::-1],
            "split": row["split"],
        }
    return processed_data


# FOR DATASET WITH UNIQUE TYPE TO CAPTURE - USED FOR COMPARING WITH WORK DONE IN THIS PAPER :
# "Advancing entity recognition in biomedicine via instruction tuning of large language models"
def process_dataset_v2(docs, df, tag2label):
    """
    Function which modifies the abstract to include the entity:
    Input : DCTN4 as a modifier of chronic Pseudomonas aeruginosa infection
    Output : @@DCTN4@@ as a modifier of @@chronic Pseudomonas aeruginosa infection@@
    -------------
    docs : dict {pmid : abstract}
    df : pd.DataFrame
    """
    processed_data = {}
    unique_values = list(set(chain.from_iterable(df["type"])))
    for pmid in docs:
        # Get the specific abstract and filter dataframe for the given document ID
        new_df = df[df["document_id"] == pmid]
        abstract = docs[pmid]

        # Sort dataframe by the start offset to handle replacements in order
        new_df["start"] = new_df["offsets"].apply(lambda x: x[0][0])
        df_sorted = new_df.sort_values(
            by="start", ascending=False
        )  # Replace from the end to avoid offset shifts

        # Perform replacements based on offsets
        spans = []
        for _, row in df_sorted.iterrows():
            start, end = row["offsets"][0]
            text = row["text"]
            label = tag2label[row["type"][0]]  # label
            entity_type = str(row["type"][0])  # type
            # print("entity_type :", entity_type)
            tagged_text = f"@@{text}@@"
            tag = row["type"][0]
            span = {
                "start": start,
                "end": end,
                "label": label,
                "tag": tag,
                "text": text,
            }
            spans.append(span)
            # Replace the text at the specific offset
            abstract = abstract[:start] + tagged_text + abstract[end:]

        processed_data[pmid] = {
            "input": docs[pmid],
            "output": abstract,
            "spans": spans[::-1],
            "split": row["split"],
        }
    return processed_data


def convert_ds_to_custom_format(ds, label2id):
    """
    Convert the Ontonotes5 Hugging Face dataset to the desired custom format (bigbio).
    Args:
        ds: The DatasetDict object (train, validation, test splits).
        label2id: Dictionary mapping entity labels to their IDs.
    Returns:
        Converted dataset as a list of dictionaries.
    """
    id2label = {v: k for k, v in label2id.items()}  # Reverse label2id for decoding tags
    converted_data = []

    # Iterate through each split in the dataset
    for split in ds.keys():  # 'train', 'validation', 'test'
        for sample in ds[split]:
            tokens = sample["tokens"]
            tags = sample["tags"]

            # Reconstruct the text
            text = " ".join(tokens)

            # Extract spans
            spans = []
            current_span = None

            for idx, tag_id in enumerate(tags):
                tag = id2label[tag_id]
                if tag.startswith("B-"):  # Beginning of a new entity
                    if current_span:  # Close the previous span
                        current_span["end"] = len(" ".join(tokens[:idx])) + idx - 1
                        current_span["text"] = text[
                            current_span["start"] : current_span["end"]
                        ]
                        spans.append(current_span)

                    # Start a new span
                    current_span = {
                        "start": len(" ".join(tokens[:idx]))
                        + idx,  # Account for spaces
                        "label": tag_id,
                        "tag": tag[2:],  # Remove "B-" prefix
                    }
                elif (
                    tag.startswith("I-")
                    and current_span
                    and current_span["tag"] == tag[2:]
                ):
                    # Continuation of the current entity
                    continue
                else:
                    # End the current span
                    if current_span:
                        current_span["end"] = len(" ".join(tokens[:idx])) + idx - 1
                        current_span["text"] = text[
                            current_span["start"] : current_span["end"]
                        ]
                        spans.append(current_span)
                        current_span = None

            # Add the last span if any
            if current_span:
                current_span["end"] = len(text)
                current_span["text"] = text[current_span["start"] : current_span["end"]]
                spans.append(current_span)

            # Add the transformed sample to the dataset
            converted_data.append(
                {
                    "text": text,
                    "spans": spans,
                    # "pmid": None,  # No pmid for OntoNotes
                    "split": split,
                }
            )

    return converted_data
