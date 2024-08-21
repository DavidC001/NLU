import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch
from transformers import AutoModel

class ModelSA(nn.Module):
    '''
        This class is used to define the model for the Sentiment Analysis task
    '''

    def __init__(self, out_sents, 
                 bert_model, dropoutBertEmb=0,
                 classification_layers=[]):
        '''
            Initialize the model

            Args:
                out_sents: Number of sentiment classes
                bert_model: BERT model to use
                dropoutBertEmb: Dropout for the representations computed by BERT, default is 0
                classification_layers: List with the number of neurons for each layer in the sentiment classification, default is []
        '''
        super(ModelSA, self).__init__()
        
        self.encoder = AutoModel.from_pretrained(bert_model)
        self.dropoutBertEmb = nn.Dropout(dropoutBertEmb)

        self.state_size = self.encoder.config.hidden_size

        classification_layers = [self.state_size] + classification_layers + [out_sents]
        layers = []
        for i, layer in enumerate(classification_layers[:-1]):
            layers.append(nn.Linear(classification_layers[i], classification_layers[i+1]))
            if i != len(classification_layers) - 2:
                layers.append(nn.ReLU())
        self.sent_out = nn.Sequential(*layers)
        
    def forward(self, utterance, attention_mask, mapping):
        '''
            Forward pass of the model

            Args:
                utterance: Tensor with the utterance
                attention_mask: Tensor with the attention mask
                mapping: Tensor with the mapping from the utterance to the slots

            Returns:
                The sentiment logits
        '''
        # utterance.size() = batch_size X seq_len
        utt_emb = self.encoder(utterance, attention_mask=attention_mask).last_hidden_state
        # breakpoint()

        # dropout
        utt_emb = self.dropoutBertEmb(utt_emb)
        
        # Compute slot logits
        slot_in = torch.stack([utt_emb[i][idx] for i, idx in enumerate(mapping)]).to(utterance.device)
        slots = self.sent_out(slot_in)
        # breakpoint()
        
        # Slot size: batch_size, seq_len, classes 
        slots = slots.permute(0,2,1) # We need this for computing the loss
        # Slot size: batch_size, classes, seq_len
        return slots

