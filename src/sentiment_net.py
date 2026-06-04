import torch
import torch.nn as nn

from src.models import GRUCell, LSTMCell, RNNCell


class SentimentModel(nn.Module):
    """
    Sentiment classifier using custom recurrent cells.

    Critical fix:
    the forward pass accepts true sequence lengths and masks recurrent state
    updates after each review's final real token. This prevents PAD tokens from
    overwriting the hidden/cell state used for classification.
    """
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        hidden_dim,
        output_dim,
        cell_type="RNN",
        n_layers=1,
        dropout=0.3,
        embedding_weights=None,
        pad_idx=0,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.cell_type = cell_type.upper()
        self.n_layers = n_layers
        self.pad_idx = pad_idx

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

        if embedding_weights is not None:
            if embedding_weights.shape != self.embedding.weight.data.shape:
                raise ValueError(
                    f"Embedding weights shape {embedding_weights.shape} does not match "
                    f"expected shape {self.embedding.weight.data.shape}"
                )
            self.embedding.weight.data.copy_(embedding_weights)

        # Ensure PAD stays zero even after copying pretrained weights.
        with torch.no_grad():
            self.embedding.weight.data[pad_idx].zero_()

        self.embedding.weight.requires_grad = True

        self.layers = nn.ModuleList()
        for layer_idx in range(n_layers):
            input_size = embedding_dim if layer_idx == 0 else hidden_dim

            if self.cell_type == "RNN":
                self.layers.append(RNNCell(input_size, hidden_dim))
            elif self.cell_type == "GRU":
                self.layers.append(GRUCell(input_size, hidden_dim))
            elif self.cell_type == "LSTM":
                self.layers.append(LSTMCell(input_size, hidden_dim))
            else:
                raise ValueError(f"Unknown cell type: {cell_type}")

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, lengths=None):
        """
        Args:
            x: token ids of shape (batch_size, seq_len)
            lengths: true non-padding lengths of shape (batch_size,)

        Returns:
            logits of shape (batch_size, output_dim)
        """
        batch_size, seq_len = x.shape
        device = x.device

        if lengths is None:
            lengths = (x != self.pad_idx).sum(dim=1).clamp(min=1)
        lengths = lengths.to(device)

        embedded = self.dropout(self.embedding(x))

        h = [
            torch.zeros(batch_size, self.hidden_dim, device=device)
            for _ in range(self.n_layers)
        ]

        if self.cell_type == "LSTM":
            c = [
                torch.zeros(batch_size, self.hidden_dim, device=device)
                for _ in range(self.n_layers)
            ]

        for t in range(seq_len):
            active = (t < lengths).float().unsqueeze(1)
            step_input = embedded[:, t, :]

            for layer_idx in range(self.n_layers):
                if self.cell_type == "LSTM":
                    h_new, c_new = self.layers[layer_idx](step_input, (h[layer_idx], c[layer_idx]))

                    h[layer_idx] = active * h_new + (1.0 - active) * h[layer_idx]
                    c[layer_idx] = active * c_new + (1.0 - active) * c[layer_idx]
                else:
                    h_new = self.layers[layer_idx](step_input, h[layer_idx])
                    h[layer_idx] = active * h_new + (1.0 - active) * h[layer_idx]

                step_input = h[layer_idx]

        final_state = h[-1]
        logits = self.fc(self.dropout(final_state))
        return logits