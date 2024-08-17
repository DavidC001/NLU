import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch

class ModelIAS(nn.Module):

    def __init__(self, hid_size, out_slot, out_int, emb_size, vocab_len, n_layer=1, pad_index=0, dropoutEmb=0, dropoutOut=0, bidirectional=False, combine='concat'):
        super(ModelIAS, self).__init__()
        # hid_size = Hidden size
        # out_slot = number of slots (output size for slot filling)
        # out_int = number of intents (output size for intent class)
        # emb_size = word embedding size
        
        self.embedding = nn.Embedding(vocab_len, emb_size, padding_idx=pad_index)
        self.dropEmb = nn.Dropout(dropoutEmb)
        self.utt_encoder = nn.LSTM(emb_size, hid_size, n_layer, bidirectional=bidirectional, batch_first=True)
        self.dropOut = nn.Dropout(dropoutOut)

        self.slot_out = nn.Linear(hid_size * (2 if bidirectional and combine=="concat" else 1), out_slot)
        self.intent_out = nn.Linear(hid_size * (2 if bidirectional and combine=="concat" else 1), out_int)

        self.bidirectional = bidirectional
        self.combination_method = combine

        if combine == "gated":
            linear = nn.Linear(hid_size * 2, hid_size)
            activation = nn.Sigmoid() 
            self.gate = nn.Sequential(linear, activation)

        # breakpoint()
        
    def forward(self, utterance, seq_lengths):
        # utterance.size() = batch_size X seq_len
        utt_emb = self.embedding(utterance) # utt_emb.size() = batch_size X seq_len X emb_size
        
        # pack_padded_sequence avoid computation over pad tokens reducing the computational cost
        
        packed_input = pack_padded_sequence(utt_emb, seq_lengths.cpu().numpy(), batch_first=True)
        # Process the batch
        packed_output, (last_hidden, cell) = self.utt_encoder(packed_input) 
       
        # Unpack the sequence
        utt_encoded, input_sizes = pad_packed_sequence(packed_output, batch_first=True)
        # Get the last hidden state
        # breakpoint()

        if not self.bidirectional:
            last_hidden = last_hidden[-1,:,:]
        else:
            if self.combination_method == "concat":
                last_hidden = torch.cat((last_hidden[0], last_hidden[1]), dim=1)
            elif self.combination_method == "sum":
                last_hidden = last_hidden[0] + last_hidden[1]
            elif self.combination_method == "gated":
                scores = self.gate(torch.cat((last_hidden[0], last_hidden[1]), dim=1))
                last_hidden = scores * last_hidden[0] + (1-scores) * last_hidden[1]
            else:
                raise ValueError("Invalid combination method")
                
        # Compute slot logits
        slots = self.slot_out(utt_encoded)
        # Compute intent logits
        intent = self.intent_out(last_hidden)
        
        # Slot size: batch_size, seq_len, classes 
        slots = slots.permute(0,2,1) # We need this for computing the loss
        # Slot size: batch_size, classes, seq_len
        return slots, intent

