import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch

class ModelIAS(nn.Module):

    def __init__(self, hid_size, out_slot, out_int, emb_size, vocab_len, n_layer=1, pad_index=0, dropoutEmb=0, dropoutOut=0, bidirectional=False, combine='concat', layerNorm=False):
        super(ModelIAS, self).__init__()
        # hid_size = Hidden size
        # out_slot = number of slots (output size for slot filling)
        # out_int = number of intents (output size for intent class)
        # emb_size = word embedding size
        
        self.embedding = nn.Embedding(vocab_len, emb_size, padding_idx=pad_index)
        self.dropEmb = nn.Dropout(dropoutEmb)
        self.utt_encoder = nn.LSTM(emb_size, hid_size, n_layer, bidirectional=bidirectional, batch_first=True)
        self.dropOut = nn.Dropout(dropoutOut)

        self.state_size = hid_size * (2 if bidirectional and combine=="concat" else 1)
        self.slot_out = nn.Linear(self.state_size, out_slot)
        self.intent_out = nn.Linear(self.state_size, out_int)

        self.bidirectional = bidirectional
        self.combination_method = combine

        if combine == "gated":
            linear = nn.Linear(hid_size * 2, hid_size)
            activation = nn.Sigmoid() 
            self.gate_intent = nn.Sequential(linear, activation)
            linear_slot = nn.Linear(hid_size * 2, hid_size)
            self.gate_slot = nn.Sequential(linear_slot, activation)

        self.layerNorm = layerNorm
        if layerNorm:
            self.ln = nn.LayerNorm(self.state_size)
            self.ln2 = nn.LayerNorm(self.state_size)

        
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
                utt_shape = utt_encoded.shape
                utt_encoded = utt_encoded.view([utt_shape[0], utt_shape[1], self.state_size, 2])
                utt_encoded = utt_encoded.sum(dim=3)
            elif self.combination_method == "gated":
                scores = self.gate_intent(torch.cat((last_hidden[0], last_hidden[1]), dim=1))
                last_hidden = scores * last_hidden[0] + (1-scores) * last_hidden[1]
                scores = self.gate_slot(utt_encoded)
                utt_shape = utt_encoded.shape
                utt_encoded = utt_encoded.view([utt_shape[0], utt_shape[1], self.state_size, 2])
                utt_encoded = scores * utt_encoded[:, :, :, 0] + (1-scores) * utt_encoded[:, :, :, 1]
                # breakpoint()
            else:
                raise ValueError("Invalid combination method")
        
        if self.layerNorm:
            last_hidden = self.ln(last_hidden)
            utt_encoded = self.ln2(utt_encoded)

        # Compute slot logits
        slots = self.slot_out(utt_encoded)
        # Compute intent logits
        intent = self.intent_out(last_hidden)
        
        # Slot size: batch_size, seq_len, classes 
        slots = slots.permute(0,2,1) # We need this for computing the loss
        # Slot size: batch_size, classes, seq_len
        return slots, intent

