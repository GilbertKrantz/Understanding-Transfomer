import torch, math
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()

        # Create a tensor filled with zeros, which will be populated with positional encodings.
        pe = torch.zeros(max_seq_length, d_model)

        # Create a A tensor containing the position indices for each position in the sequence.
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)

        #  A term used to scale the position indices in a specific way.
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]