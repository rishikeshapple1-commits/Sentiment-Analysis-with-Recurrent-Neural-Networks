import re
from collections import Counter

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


CONTRACTIONS = {
    "can't": "can not",
    "cannot": "can not",
    "won't": "will not",
    "n't": " not",
    "i'm": "i am",
    "it's": "it is",
    "he's": "he is",
    "she's": "she is",
    "that's": "that is",
    "there's": "there is",
    "what's": "what is",
    "you're": "you are",
    "we're": "we are",
    "they're": "they are",
    "i've": "i have",
    "we've": "we have",
    "they've": "they have",
    "i'd": "i would",
    "you'd": "you would",
    "i'll": "i will",
    "you'll": "you will",
}


def clean_text(text):
    """
    Clean IMDB review text while preserving sentiment-critical negation.

    Important fix compared with the earlier version:
    contractions such as "didn't" become "did not" instead of losing the
    negation information during punctuation removal.
    """
    text = text.lower()
    text = re.sub(r"<br\s*/?>", " ", text)

    for src, dst in CONTRACTIONS.items():
        text = text.replace(src, dst)

    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def create_vocab(dataset_split, max_size=10000):
    """
    Build a vocabulary from the training split only.
    Index 0 is reserved for padding; index 1 is reserved for unknown words.
    """
    counter = Counter()

    for i in tqdm(range(len(dataset_split)), desc="Building vocab"):
        tokens = clean_text(dataset_split[i]["text"]).split()
        counter.update(tokens)

    vocab = {"<PAD>": 0, "<UNK>": 1}
    for word, _ in counter.most_common(max_size - 2):
        vocab[word] = len(vocab)

    return vocab


def load_glove_embeddings(path, vocab, embedding_dim):
    """
    Load pretrained GloVe vectors into a vocabulary-aligned matrix.

    Critical fix:
    the PAD row is forced to an all-zero vector. In the earlier implementation,
    words not found in GloVe were randomly initialized, which also affected
    <PAD>. That allowed padding tokens to corrupt the final recurrent state.
    """
    embeddings_dict = {}
    print(f"Loading GloVe embeddings from {path}...")

    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                values = line.rstrip().split()
                if len(values) != embedding_dim + 1:
                    continue

                word = values[0]
                vector = np.asarray(values[1:], dtype="float32")
                embeddings_dict[word] = vector
    except FileNotFoundError:
        print(f"Warning: GloVe file not found at {path}. Using random initialization.")
        return None

    weights_matrix = np.random.normal(
        loc=0.0,
        scale=0.05,
        size=(len(vocab), embedding_dim),
    ).astype("float32")

    # Padding must be exactly zero and remain non-trainable through padding_idx.
    weights_matrix[vocab["<PAD>"]] = np.zeros(embedding_dim, dtype="float32")

    words_found = 0
    for word, idx in vocab.items():
        if word in embeddings_dict:
            weights_matrix[idx] = embeddings_dict[word]
            words_found += 1

    print(f"Successfully loaded {words_found}/{len(vocab)} words from GloVe.")
    return torch.tensor(weights_matrix, dtype=torch.float32)


def encode_text(text, vocab, max_length, truncation="head"):
    """
    Convert raw text to token ids, true sequence length, and padded tensor.

    truncation options:
      - "head": keep first max_length tokens
      - "head_tail": keep beginning and ending, useful for late negation tests
    """
    tokens = clean_text(text).split()
    encoded = [vocab.get(token, vocab["<UNK>"]) for token in tokens]

    if len(encoded) > max_length:
        if truncation == "head_tail" and max_length >= 2:
            first_part = max_length // 2
            second_part = max_length - first_part
            encoded = encoded[:first_part] + encoded[-second_part:]
        else:
            encoded = encoded[:max_length]

    true_length = max(1, min(len(encoded), max_length))

    if len(encoded) < max_length:
        encoded = encoded + [vocab["<PAD>"]] * (max_length - len(encoded))

    return torch.tensor(encoded, dtype=torch.long), torch.tensor(true_length, dtype=torch.long)


class IMDBDataset(Dataset):
    def __init__(self, data, vocab, max_length=200, truncation="head"):
        self.data = data
        self.vocab = vocab
        self.max_length = max_length
        self.truncation = truncation

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        x, length = encode_text(
            item["text"],
            self.vocab,
            self.max_length,
            truncation=self.truncation,
        )
        y = torch.tensor(item["label"], dtype=torch.long)
        return x, length, y


def get_dataloaders(
    batch_size=64,
    max_length=200,
    vocab_size=10000,
    val_ratio=0.10,
    seed=42,
    num_workers=0,
    truncation="head",
):
    """
    Load IMDB and return train/validation/test loaders.

    Critical fix:
    the test split is no longer used as validation during training. We create a
    validation split from the original training data and reserve the official
    IMDB test split for final evaluation only.
    """
    dataset = load_dataset("imdb")

    split = dataset["train"].train_test_split(
        test_size=val_ratio,
        seed=seed,
        stratify_by_column="label",
    )

    train_raw = split["train"]
    val_raw = split["test"]
    test_raw = dataset["test"]

    vocab = create_vocab(train_raw, max_size=vocab_size)

    train_ds = IMDBDataset(train_raw, vocab, max_length=max_length, truncation=truncation)
    val_ds = IMDBDataset(val_raw, vocab, max_length=max_length, truncation=truncation)
    test_ds = IMDBDataset(test_raw, vocab, max_length=max_length, truncation=truncation)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    raw_splits = {
        "train": train_raw,
        "val": val_raw,
        "test": test_raw,
    }

    return train_loader, val_loader, test_loader, vocab, raw_splits