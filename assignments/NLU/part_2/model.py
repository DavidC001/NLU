import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch
from transformers import AutoModel

class ModelIAS(nn.Module):

    def __init__(self, out_slot, out_int, 
                 bert_model, pad_index, dropoutBertEmb=0,
                 classification_layers_slots=[], classification_layers_intents=[]):
        super(ModelIAS, self).__init__()
        # hid_size = Hidden size
        # out_slot = number of slots (output size for slot filling)
        # out_int = number of intents (output size for intent class)
        # emb_size = word embedding size
        
        self.encoder = AutoModel.from_pretrained(bert_model)
        self.dropoutBertEmb = nn.Dropout(dropoutBertEmb)

        self.state_size = self.encoder.config.hidden_size

        classification_layers_slots = [self.state_size] + classification_layers_slots + [out_slot]
        layers = []
        for i, layer in enumerate(classification_layers_slots[:-1]):
            layers.append(nn.Linear(classification_layers_slots[i], classification_layers_slots[i+1]))
            if i != len(classification_layers_slots) - 2:
                layers.append(nn.ReLU())
        self.slot_out = nn.Sequential(*layers)

        classification_layers_intents = [self.state_size] + classification_layers_intents + [out_int]
        layers = []
        for i, layer in enumerate(classification_layers_intents[:-1]):
            layers.append(nn.Linear(classification_layers_intents[i], classification_layers_intents[i+1]))
            if i != len(classification_layers_intents) - 2:
                layers.append(nn.ReLU())
        self.intent_out = nn.Sequential(*layers)

        # breakpoint()
        
    def forward(self, utterance, attention_mask, mapping):
        # utterance.size() = batch_size X seq_len
        utt_emb = self.encoder(utterance, attention_mask=attention_mask).last_hidden_state
        # breakpoint()

        # dropout
        utt_emb = self.dropoutBertEmb(utt_emb)
        
        # Compute slot logits
        slot_in = torch.stack([utt_emb[i][idx] for i, idx in enumerate(mapping)]).to(utterance.device)
        slots = self.slot_out(slot_in)
        # Compute intent logits
        intent = self.intent_out(utt_emb[:,0,:])
        # breakpoint()
        
        # Slot size: batch_size, seq_len, classes 
        slots = slots.permute(0,2,1) # We need this for computing the loss
        # Slot size: batch_size, classes, seq_len
        return slots, intent

