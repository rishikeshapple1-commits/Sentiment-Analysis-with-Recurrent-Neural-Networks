#!/usr/bin/env python3
"""
Full training script for sentiment analysis with custom RNN/GRU/LSTM cells.
"""
import sys

sys.path.insert(0, ".")

from src.train import train_and_compare


if __name__ == "__main__":
    print("=" * 60)
    print("SENTIMENT ANALYSIS: RNN vs GRU vs LSTM Comparison")
    print("=" * 60)
    print()

    comparison_stats = train_and_compare(
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
    )

    print("\n" + "=" * 60)
    print("FINAL RESULTS SUMMARY")
    print("=" * 60)

    for cell, stats in comparison_stats.items():
        late = stats["late_negation"]
        late_acc = late["accuracy"] if late["accuracy"] is not None else "N/A"

        if isinstance(late_acc, float):
            late_acc_str = f"{late_acc:.2f}%"
        else:
            late_acc_str = str(late_acc)

        print(f"\n{cell} Model:")
        print(f"  Best Validation Accuracy: {stats['best_val_accuracy']:.2f}%")
        print(f"  Test Accuracy: {stats['test_accuracy']:.2f}%")
        print(f"  Test Loss: {stats['test_loss']:.4f}")
        print(f"  Training Time: {stats['training_time']:.2f}s")
        print(f"  Final Training Loss: {stats['train_losses'][-1]:.4f}")
        print(f"  Late Negation Examples: {late['num_examples']}")
        print(f"  Late Negation Accuracy: {late_acc_str}")
        print(f"  Sequence Length Sensitivity: {stats['sensitivity']}")

    print("\nComparison results saved to: comparison_results.png")