from torch import nn

class LM_RNN(nn.Module):
    """
    Language Model RNN class.

    Args:
        emb_size (int): The size of the embedding layer.
        hidden_size (int): The size of the hidden layer.
        output_size (int): The size of the output layer (vocabulary size).
        pad_index (int, optional): The index used for padding. Defaults to 0.
        n_layers (int, optional): The number of layers in the RNN. Defaults to 1.
    """

    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, n_layers=1):
        super(LM_RNN, self).__init__()
        # Token ids to vectors, we will better see this in the next lab
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        # Pytorch's RNN layer: https://pytorch.org/docs/stable/generated/torch.nn.RNN.html
        self.rnn = nn.RNN(emb_size, hidden_size, n_layers, bidirectional=False, batch_first=True)
        self.pad_token = pad_index
        # Linear layer to project the hidden layer to our output space
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, input_sequence):
        emb = self.embedding(input_sequence)
        rnn_out, _  = self.rnn(emb)
        output = self.output(rnn_out).permute(0,2,1)
        return output

class LM_LSTM(nn.Module):
    """
    A language model LSTM class.

    Args:
        emb_size (int): The size of the embedding layer.
        hidden_size (int): The size of the hidden layer.
        output_size (int): The size of the output layer.
        pad_index (int, optional): The index used for padding. Defaults to 0.
        out_dropout (float, optional): The dropout rate before the output layer. Defaults to 0.
        emb_dropout (float, optional): The dropout rate after the embedding layer. Defaults to 0.
        n_layers (int, optional): The number of layers in the LSTM. Defaults to 1.
    """

    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, out_dropout=0, emb_dropout=0, n_layers=1):
        super(LM_LSTM, self).__init__()
        # Token ids to vectors, we will better see this in the next lab
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        # Pytorch's RNN layer: https://pytorch.org/docs/stable/generated/torch.nn.RNN.html
        self.rnn = nn.LSTM(emb_size, hidden_size, n_layers, bidirectional=False, batch_first=True)
        self.pad_token = pad_index
        # Linear layer to project the hidden layer to our output space
        self.output = nn.Linear(hidden_size, output_size)
        # Dropout layers
        self.out_dropout = nn.Dropout(out_dropout)
        self.emb_dropout = nn.Dropout(emb_dropout)

    def forward(self, input_sequence):
        emb = self.embedding(input_sequence)
        emb_drop = self.emb_dropout(emb)
        rnn_out, _  = self.rnn(emb_drop)
        rnn_out_drop = self.out_dropout(rnn_out)
        output = self.output(rnn_out_drop).permute(0,2,1)
        return output
