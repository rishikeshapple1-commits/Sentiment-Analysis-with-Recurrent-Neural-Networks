import copy
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from torch.utils.data import DataLoader, Subset

from src.data_loader import (
    IMDBDataset,
    clean_text,
    encode_text,
    get_dataloaders,
    load_glove_embeddings,
)
from src.late_negation_analysis import get_late_negation_indices
from src.sentiment_net import SentimentModel


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def build_model(
    vocab_size,
    embedding_dim,
    hidden_dim,
    output_dim,
    cell_type,
    embedding_weights=None,
    n_layers=1,
    dropout=0.3,
):
    return SentimentModel(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        cell_type=cell_type,
        n_layers=n_layers,
        dropout=dropout,
        embedding_weights=embedding_weights,
    )


def train_one_epoch(model, dataloader, device, criterion, optimizer, max_grad_norm=1.0):
    model.train()

    total_loss = 0.0
    all_preds = []
    all_labels = []

    for texts, lengths, labels in dataloader:
        texts = texts.to(device)
        lengths = lengths.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(texts, lengths)
        loss = criterion(outputs, labels)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)

        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())

    avg_loss = total_loss / max(len(dataloader), 1)
    acc = accuracy_score(all_labels, all_preds) if all_labels else 0.0
    return avg_loss, acc


def evaluate_model(model, dataloader, device, criterion=None):
    model.eval()

    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for texts, lengths, labels in dataloader:
            texts = texts.to(device)
            lengths = lengths.to(device)
            labels = labels.to(device)

            outputs = model(texts, lengths)

            if criterion is not None:
                loss = criterion(outputs, labels)
                total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / max(len(dataloader), 1) if criterion is not None else 0.0
    acc = accuracy_score(all_labels, all_preds) if all_labels else 0.0
    return avg_loss, acc, all_labels, all_preds


def make_subset_loader(dataset_split, indices, vocab, batch_size, max_length, truncation="head_tail"):
    subset_dataset = IMDBDataset(
        dataset_split.select(indices),
        vocab,
        max_length=max_length,
        truncation=truncation,
    )
    return DataLoader(subset_dataset, batch_size=batch_size, shuffle=False)


def analyze_late_negation(model, test_raw, vocab, device, batch_size=64, max_length=300, max_items=1000):
    indices = get_late_negation_indices(test_raw, max_items=max_items)

    if not indices:
        return {
            "num_examples": 0,
            "accuracy": None,
        }

    loader = make_subset_loader(
        test_raw,
        indices,
        vocab,
        batch_size=batch_size,
        max_length=max_length,
        truncation="head_tail",
    )

    _, acc, _, _ = evaluate_model(model, loader, device, criterion=None)
    return {
        "num_examples": len(indices),
        "accuracy": acc * 100.0,
    }


def analyze_sequence_length_sensitivity(
    model,
    test_raw,
    vocab,
    device,
    lengths=None,
    sample_size=1000,
    seed=42,
):
    """
    Evaluate the same trained model using different input truncation lengths.
    This tests how much information each model uses from longer reviews.
    """
    if lengths is None:
        lengths = [50, 100, 200, 300, 500]

    rng = random.Random(seed)
    indices = list(range(len(test_raw)))
    rng.shuffle(indices)
    indices = indices[:min(sample_size, len(indices))]

    results = {}

    for max_length in lengths:
        correct = 0
        total = 0

        for idx in indices:
            item = test_raw[idx]
            x, seq_len = encode_text(
                item["text"],
                vocab,
                max_length=max_length,
                truncation="head",
            )

            x = x.unsqueeze(0).to(device)
            seq_len = seq_len.unsqueeze(0).to(device)
            label = int(item["label"])

            with torch.no_grad():
                output = model(x, seq_len)
                pred = torch.argmax(output, dim=1).item()

            correct += int(pred == label)
            total += 1

        results[max_length] = correct / total if total else 0.0

    return results


def train_single_model(
    cell,
    train_loader,
    val_loader,
    test_loader,
    raw_splits,
    vocab,
    embedding_weights,
    device,
    epochs=5,
    lr=0.001,
    hidden_dim=128,
    n_layers=1,
    dropout=0.3,
    batch_size=64,
):
    print(f"\n{'=' * 20}")
    print(f"Training {cell} Model")
    print(f"{'=' * 20}")

    model = build_model(
        vocab_size=len(vocab),
        embedding_dim=100,
        hidden_dim=hidden_dim,
        output_dim=2,
        cell_type=cell,
        embedding_weights=embedding_weights,
        n_layers=n_layers,
        dropout=dropout,
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    # Same optimizer and learning rate for all models for a fair comparison.
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    best_val_acc = -1.0
    best_state = copy.deepcopy(model.state_dict())

    start_time = time.time()

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            device,
            criterion,
            optimizer,
        )
        val_loss, val_acc, _, _ = evaluate_model(model, val_loader, device, criterion)

        train_losses.append(train_loss)
        train_accuracies.append(train_acc * 100.0)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc * 100.0)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())

        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Loss: {train_loss:.4f} "
            f"Acc: {train_acc * 100.0:.2f}% "
            f"Val Loss: {val_loss:.4f} "
            f"Val Acc: {val_acc * 100.0:.2f}%"
        )

    training_time = time.time() - start_time
    model.load_state_dict(best_state)

    test_loss, test_acc, test_labels, test_preds = evaluate_model(
        model,
        test_loader,
        device,
        criterion,
    )

    model_path = f"{cell.lower()}_model.pt"
    torch.save(model.state_dict(), model_path)

    print(f"Finished {cell} in {training_time:.2f}s")
    print(f"Best Val Accuracy: {best_val_acc * 100.0:.2f}%")
    print(f"Test Loss: {test_loss:.4f} Test Accuracy: {test_acc * 100.0:.2f}%")
    print(f"Saved best {cell} model to {model_path}")

    sensitivity = analyze_sequence_length_sensitivity(
        model,
        raw_splits["test"],
        vocab,
        device,
    )

    late_negation = analyze_late_negation(
        model,
        raw_splits["test"],
        vocab,
        device,
        batch_size=batch_size,
        max_length=300,
    )

    cm = confusion_matrix(test_labels, test_preds).tolist()
    report = classification_report(
        test_labels,
        test_preds,
        target_names=["negative", "positive"],
        output_dict=True,
        zero_division=0,
    )

    return model, {
        "train_losses": train_losses,
        "train_accuracies": train_accuracies,
        "val_losses": val_losses,
        "val_accuracies": val_accuracies,
        "best_val_accuracy": best_val_acc * 100.0,
        "test_accuracy": test_acc * 100.0,
        "test_loss": test_loss,
        "training_time": training_time,
        "sensitivity": sensitivity,
        "late_negation": late_negation,
        "confusion_matrix": cm,
        "classification_report": report,
    }


def plot_comparison_results(comparison_stats, output_path="comparison_results.png"):
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    for cell, stats in comparison_stats.items():
        plt.plot(stats["train_losses"], marker="o", label=f"{cell} train")
        plt.plot(stats["val_losses"], marker="x", linestyle="--", label=f"{cell} val")
    plt.title("Loss Comparison")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 3, 2)
    for cell, stats in comparison_stats.items():
        plt.plot(stats["train_accuracies"], marker="o", label=f"{cell} train")
        plt.plot(stats["val_accuracies"], marker="x", linestyle="--", label=f"{cell} val")
    plt.title("Accuracy Comparison")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()

    plt.subplot(1, 3, 3)
    for cell, stats in comparison_stats.items():
        lengths = sorted(stats["sensitivity"].keys())
        accs = [stats["sensitivity"][length] * 100.0 for length in lengths]
        plt.plot(lengths, accs, marker="o", label=cell)
    plt.title("Sequence Length Sensitivity")
    plt.xlabel("Input length")
    plt.ylabel("Accuracy (%)")
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    print(f"Comparison plot saved to {output_path}")


def train_and_compare(
    epochs=5,
    lr=0.001,
    batch_size=64,
    max_length=200,
    use_glove=True,
    hidden_dim=128,
    n_layers=1,
    dropout=0.3,
    vocab_size=10000,
    seed=42,
):
    set_seed(seed)
    device = get_device()

    print("Loading dataset and preparing loaders...")
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader, vocab, raw_splits = get_dataloaders(
        batch_size=batch_size,
        max_length=max_length,
        vocab_size=vocab_size,
        seed=seed,
        truncation="head",
    )

    embedding_weights = None
    if use_glove:
        glove_path = "data/glove.6B.100d.txt"
        embedding_weights = load_glove_embeddings(glove_path, vocab, 100)

    comparison_stats = {}

    for cell in ["RNN", "GRU", "LSTM"]:
        _, stats = train_single_model(
            cell=cell,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            raw_splits=raw_splits,
            vocab=vocab,
            embedding_weights=embedding_weights,
            device=device,
            epochs=epochs,
            lr=lr,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            dropout=dropout,
            batch_size=batch_size,
        )
        comparison_stats[cell] = stats

    plot_comparison_results(comparison_stats)

    return comparison_stats


if __name__ == "__main__":
    train_and_compare()