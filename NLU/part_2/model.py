import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch
from transformers import AutoModel

class ModelIAS(nn.Module):
    '''
        This class is used to define the model for the Intent and Slot Filling task
    '''

    def __init__(self, out_slot, out_int, 
                 bert_model, dropoutBertEmb=0,
                 classification_layers_slots=[], classification_layers_intents=[]):
        '''
            Initialize the model

            Args:
                out_slot: Number of slot classes
                out_int: Number of intent classes
                bert_model: BERT model to use
                dropoutBertEmb: Dropout for the representations computed by BERT, default is 0
                classification_layers_slots: List with the number of neurons for each layer in the slot classification, default is []
                classification_layers_intents: List with the number of neurons for each layer in the intent classification, default is []
        '''
        super(ModelIAS, self).__init__()

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
        
    def forward(self, utterance, attention_mask, mapping):
        '''
            Forward pass of the model

            Args:
                utterance: Tensor with the utterance
                attention_mask: Tensor with the attention mask
                mapping: Tensor with the mapping from the utterance to the slots

            Returns:
                The slot and intent logits
        '''
        # Compute token representations with BERT
        utt_emb = self.encoder(utterance, attention_mask=attention_mask).last_hidden_state
        # dropout
        utt_emb = self.dropoutBertEmb(utt_emb)
        
        # Compute slot logits
        slot_in = torch.stack([utt_emb[i][idx] for i, idx in enumerate(mapping)]).to(utterance.device)
        slots = self.slot_out(slot_in)
        # Compute intent logits using the [CLS] token
        intent = self.intent_out(utt_emb[:,0,:])
        
        # Slot size: batch_size, seq_len, classes 
        slots = slots.permute(0,2,1) # We need this for computing the loss
        # Slot size: batch_size, classes, seq_len
        return slots, intent

