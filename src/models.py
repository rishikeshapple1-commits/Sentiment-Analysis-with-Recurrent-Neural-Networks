import torch
import torch.nn as nn
import math


class RNNCell(nn.Module):
    """
    Standard Elman RNN cell implemented from scratch.

    h_t = tanh(x_t W_ih + b_ih + h_{t-1} W_hh + b_hh)
    """
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.weight_ih = nn.Parameter(torch.empty(input_size, hidden_size))
        self.weight_hh = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.bias_ih = nn.Parameter(torch.zeros(hidden_size))
        self.bias_hh = nn.Parameter(torch.zeros(hidden_size))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight_ih)
        nn.init.orthogonal_(self.weight_hh)
        nn.init.zeros_(self.bias_ih)
        nn.init.zeros_(self.bias_hh)

    def forward(self, x, h):
        return torch.tanh(
            torch.matmul(x, self.weight_ih) + self.bias_ih +
            torch.matmul(h, self.weight_hh) + self.bias_hh
        )


class GRUCell(nn.Module):
    """
    Gated Recurrent Unit cell implemented from scratch.

    Gates:
      r_t = reset gate
      z_t = update gate
      n_t = candidate hidden state

    h_t = z_t * h_{t-1} + (1 - z_t) * n_t
    """
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.w_ih = nn.Parameter(torch.empty(input_size, 3 * hidden_size))
        self.w_hh = nn.Parameter(torch.empty(hidden_size, 3 * hidden_size))
        self.b_ih = nn.Parameter(torch.zeros(3 * hidden_size))
        self.b_hh = nn.Parameter(torch.zeros(3 * hidden_size))

        self.reset_parameters()

    def reset_parameters(self):
        for gate in range(3):
            start = gate * self.hidden_size
            end = (gate + 1) * self.hidden_size
            nn.init.xavier_uniform_(self.w_ih[:, start:end])
            nn.init.orthogonal_(self.w_hh[:, start:end])

        nn.init.zeros_(self.b_ih)
        nn.init.zeros_(self.b_hh)

    def forward(self, x, h):
        gates_i = torch.matmul(x, self.w_ih) + self.b_ih
        gates_h = torch.matmul(h, self.w_hh) + self.b_hh

        i_r, i_z, i_n = gates_i.chunk(3, dim=1)
        h_r, h_z, h_n = gates_h.chunk(3, dim=1)

        reset_gate = torch.sigmoid(i_r + h_r)
        update_gate = torch.sigmoid(i_z + h_z)
        new_gate = torch.tanh(i_n + reset_gate * h_n)

        h_next = update_gate * h + (1.0 - update_gate) * new_gate
        return h_next


class LSTMCell(nn.Module):
    """
    Long Short-Term Memory cell implemented from scratch.

    Gates:
      i_t = input gate
      f_t = forget gate
      g_t = candidate cell state
      o_t = output gate

    c_t = f_t * c_{t-1} + i_t * g_t
    h_t = o_t * tanh(c_t)
    """
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.w_ih = nn.Parameter(torch.empty(input_size, 4 * hidden_size))
        self.w_hh = nn.Parameter(torch.empty(hidden_size, 4 * hidden_size))
        self.b_ih = nn.Parameter(torch.zeros(4 * hidden_size))
        self.b_hh = nn.Parameter(torch.zeros(4 * hidden_size))

        self.reset_parameters()

    def reset_parameters(self):
        for gate in range(4):
            start = gate * self.hidden_size
            end = (gate + 1) * self.hidden_size
            nn.init.xavier_uniform_(self.w_ih[:, start:end])
            nn.init.orthogonal_(self.w_hh[:, start:end])

        nn.init.zeros_(self.b_ih)
        nn.init.zeros_(self.b_hh)

        # Gate order is input, forget, cell/candidate, output.
        # A positive forget bias improves early LSTM training stability.
        self.b_ih.data[self.hidden_size:2 * self.hidden_size].fill_(1.0)

    def forward(self, x, states):
        h, c = states

        gates_i = torch.matmul(x, self.w_ih) + self.b_ih
        gates_h = torch.matmul(h, self.w_hh) + self.b_hh

        i_i, i_f, i_g, i_o = gates_i.chunk(4, dim=1)
        h_i, h_f, h_g, h_o = gates_h.chunk(4, dim=1)

        input_gate = torch.sigmoid(i_i + h_i)
        forget_gate = torch.sigmoid(i_f + h_f)
        cell_gate = torch.tanh(i_g + h_g)
        output_gate = torch.sigmoid(i_o + h_o)

        c_next = forget_gate * c + input_gate * cell_gate
        h_next = output_gate * torch.tanh(c_next)

        return h_next, c_next