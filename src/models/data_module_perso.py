from typing import List, Dict, Union
from transformers import PreTrainedTokenizer
from torch.utils.data import Dataset
import torch


class DatasetNER(Dataset):
    def __init__(
        self,
        data: List[Dict],
        tokenizer: PreTrainedTokenizer,
        tags: List[str],
        label2id: Dict[str, int],
        max_length: int = 512,
        debug=False,
        split: str = "train",
    ):
        self.data = [entry for entry in data if entry["split"] == split]
        self.tokenizer = tokenizer
        self.tags = tags
        self.label2id = label2id
        self.max_length = max_length
        self.debug = debug
        self.inputs = []
        self.labels = []
        self.indices = (
            []
        )  # (indice, num_chunks) of all text with num_chunks > 1 (sequence_length > max_length = 512 for BERT)

        self._prepare_dataset()

    def _prepare_dataset(self):
        counter = 0
        if self.debug:
            self.data = self.data[:20]
        for idx, entry in enumerate(self.data):
            text = entry["text"]
            spans = entry.get("spans", [])

            # Tokenize the text with overflow handling.
            # The stride (here set to 50) can be adjusted or removed if you don't want overlapping tokens.
            tokenized = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_length,
                # stride=50,  # Optional: adjust for overlap if desired
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
            )

            num_chunks = len(tokenized["input_ids"])
            if num_chunks > 1:
                self.indices.append(
                    (counter, num_chunks)
                )  # Store the index of the original entry
                # print(f"Entry {counter} has {num_chunks} chunks.")
            counter += num_chunks
            for i in range(num_chunks):
                offsets = tokenized["offset_mapping"][i]
                input_ids = tokenized["input_ids"][i]
                attention_mask = tokenized["attention_mask"][i]
                offset_mapping = tokenized["offset_mapping"][i]
                tokens = self.tokenizer.convert_ids_to_tokens(input_ids)

                # Initialize labels with 'O' for each token in the chunk.
                labels = ["O"] * len(input_ids)
                for span in spans:
                    span_start, span_end, tag = (
                        span["start"],
                        span["end"],
                        span["tag"],
                    )
                    if tag not in self.tags:
                        continue
                    for j, (start, end) in enumerate(offset_mapping):
                        if start >= span_start and end <= span_end:
                            if labels[j] == "O":  # assign only if not labeled yet
                                if start == span_start:
                                    labels[j] = f"B-{tag}"
                                else:
                                    labels[j] = f"I-{tag}"

                # Assign -100 to special tokens.
                for k, (start, end) in enumerate(offset_mapping):
                    if start == end:
                        labels[k] = -100

                # Convert labels to numerical IDs.
                label_ids = [
                    self.label2id[label] if isinstance(label, str) else -100
                    for label in labels
                ]

                self.inputs.append(
                    {
                        "text": text,
                        "tokenized_text": tokens,
                        "offset_mapping": offset_mapping,
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                    }
                )
                self.labels.append(label_ids)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        item = self.inputs[idx].copy()
        item["labels"] = self.labels[idx]
        output = {}
        for key, val in item.items():
            # Only convert numeric fields to tensor.
            if key in ["input_ids", "attention_mask", "labels"]:
                output[key] = torch.tensor(val)
            else:
                output[key] = val  # e.g., tokens remain as a list of strings
        return output

    def select(self, indices):
        """Return a new dataset with only selected indices."""
        selected_data = [self[i] for i in indices]  # Select elements by index
        return DatasetNER.from_list(selected_data)  # Convert back to DatasetNER


def create_label2id(data: List[Dict], tags: List) -> Dict[str, int]:
    """
    Automatically create label2id mapping from dataset.

    Args:
        data (List[Dict]): The input dataset in the provided format.
        tags (List): List of valid tags.

    Returns:
        Dict[str, int]: Mapping of unique labels to numerical IDs.
    """
    unique_labels = {"O"}  # Start with "O" for outside tokens
    for entry in data:
        spans = entry.get("spans", [])
        for span in spans:
            tag = span["tag"]
            if tag not in tags:
                continue
            unique_labels.add(f"B-{tag}")  # Beginning tag
            unique_labels.add(f"I-{tag}")  # Inside tag

    # Create label2id dictionary sorted alphabetically for consistent order
    return {label: idx for idx, label in enumerate(sorted(unique_labels))}
