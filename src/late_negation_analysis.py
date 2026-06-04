import re


NEGATION_WORDS = {
    "not",
    "never",
    "no",
    "nothing",
    "nowhere",
    "neither",
    "nor",
    "hardly",
    "barely",
    "scarcely",
}


CONTRACTION_REPLACEMENTS = {
    "can't": "can not",
    "cannot": "can not",
    "won't": "will not",
    "n't": " not",
}


def tokenize_for_negation(text):
    text = text.lower()
    text = re.sub(r"<br\s*/?>", " ", text)

    for src, dst in CONTRACTION_REPLACEMENTS.items():
        text = text.replace(src, dst)

    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return text.split()


def contains_late_negation(text, final_fraction=1 / 3, min_tokens=10):
    """
    Return True if a negation cue appears in the final part of the review.
    By default, this checks the final third of the token sequence.
    """
    tokens = tokenize_for_negation(text)

    if len(tokens) < min_tokens:
        return False

    start = int(len(tokens) * (1.0 - final_fraction))
    tail_tokens = tokens[start:]

    return any(token in NEGATION_WORDS for token in tail_tokens)


def get_late_negation_indices(dataset_split, max_items=None):
    """
    Return dataset indices containing late negation.
    """
    indices = []

    for idx, item in enumerate(dataset_split):
        if contains_late_negation(item["text"]):
            indices.append(idx)

        if max_items is not None and len(indices) >= max_items:
            break

    return indices