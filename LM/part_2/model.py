from torch import nn
import torch

class VariationalDropout(nn.Module):
    """
    Variational Dropout module.

    This module applies variational dropout to the input tensor during training.
    Variational dropout randomly sets elements of the input tensor to zero with a probability of `dropout_probability`.
    The dropout mask is shared across the input sequence and is different for each batch.

    Args:
        dropout_probability (float): The probability of setting elements to zero. Should be in the range [0, 1].

    Returns:
        torch.Tensor: The input tensor after applying variational dropout.

    """
    def __init__(self, dropout_probability: float,):
        super().__init__()
        self.p = dropout_probability

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p <= 0.:
            return x

        batch_size = x.size(0)

        mask = x.new_empty(batch_size, 1, x.size(2), requires_grad=False).bernoulli_(1 - self.p)

        mask = mask.expand_as(x)
        x = x.mul(mask).div(1.0 - self.p)
        
        return x

class LM_LSTM_WT_VD(nn.Module):
    """
    A language model LSTM with weight tying and variational dropout.

    Args:
        emb_size (int): The size of the embedding layer.
        hidden_size (int): The size of the hidden layer.
        output_size (int): The size of the output layer.
        pad_index (int, optional): The index of the padding token. Defaults to 0.
        out_dropout (float, optional): The dropout rate before the output layer. Defaults to 0.
        emb_dropout (float, optional): The dropout rate after the embedding layer. Defaults to 0.
        n_layers (int, optional): The number of LSTM layers. Defaults to 1.
    """
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, 
                 out_dropout=0, emb_dropout=0, n_layers=1):
        super(LM_LSTM_WT_VD, self).__init__()
        # Token ids to vectors, we will better see this in the next lab
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        self.rnn = nn.LSTM(emb_size, hidden_size, n_layers, bidirectional=False, batch_first=True)
        self.pad_token = pad_index
        # Linear layer to project the hidden layer to our output space
        self.output = nn.Linear(hidden_size, output_size)
        # weight tying
        if emb_size != hidden_size: raise ValueError("emb_size and hidden_size must be equal for weight tying")
        self.output.weight = self.embedding.weight
        # variational dropout
        self.emb_dropout = VariationalDropout(emb_dropout)
        # variational dropout
        self.out_dropout = VariationalDropout(out_dropout)


    def forward(self, input_sequence):
        emb = self.embedding(input_sequence)
        emb_drop = self.emb_dropout(emb)
        rnn_out, _  = self.rnn(emb_drop)
        out_drop = self.out_dropout(rnn_out)
        output = self.output(out_drop).permute(0,2,1)
        return output
