from collections import Counter
import torch
import torch.utils.data as data
import json
import os
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

class Lang():
    '''
        Class to map words, intents and slots to integers
    '''
    def __init__(self, intents, slots, bert_model):
        '''
            Initialize the Lang class with the words, intents and slots

            Args:
                intents : list of intents
                slots : list of slots
                bert_model : BERT model to use for tokenization
        '''
        # self.word2id = self.w2id(words, cutoff=cutoff, unk=True)
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model)
        self.pad_token = self.tokenizer.pad_token_id
        self.label_pad = 0
        self.slot2id = self.lab2id(slots)
        self.intent2id = self.lab2id(intents, pad=False)
        # self.id2word = {v:k for k, v in self.word2id.items()}
        self.id2slot = {v:k for k, v in self.slot2id.items()}
        self.id2intent = {v:k for k, v in self.intent2id.items()}
    
    def lab2id(self, elements, pad=True):
        '''
            Map labels to integers

            Args:
                elements : list of labels
                pad : whether to include the padding token

            Returns:
                vocab : dictionary mapping labels to integers
        '''
        vocab = {}
        if pad:
            vocab['pad'] = 0
        for elem in elements:
                vocab[elem] = len(vocab)
        return vocab

class IntentsAndSlots (data.Dataset):
    '''
        Class to load the dataset
    '''
    
    def __init__(self, dataset, lang):
        '''
            Initialize the dataset class

            Args:
                dataset : list of samples
                lang : Lang object
        '''
        self.utterances = []
        self.intents = []
        self.slots = []
        self.tokenizer = lang.tokenizer
        
        for x in dataset:
            self.utterances.append(x['utterance'])
            self.slots.append(x['slots'])
            self.intents.append(x['intent'])

        self.slot_ids = self.mapping_seq(self.slots, lang.slot2id)
        self.intent_ids = self.mapping_lab(self.intents, lang.intent2id)

    def __len__(self):
        '''
            Return the length of the dataset

            Returns:
                length of the dataset
        '''
        return len(self.utterances)

    def __getitem__(self, idx):
        '''
            Get an item from the dataset

            Args:
                idx : index of the item

            Returns:
                sample : dictionary with the utterance, slots and intent with the following keys:
                    - utterance : tensor with the utterance tokens
                    - attention : tensor with the attention mask
                    - mapping : tensor with the mapping from the utterance to the slots
                    - sentence : list with the words in the sentence
                    - slots : tensor with the slots
                    - intent : tensor with the intent
        '''
        tokenized = self.tokenizer(self.utterances[idx], return_tensors='pt')
        utt = tokenized['input_ids'][0]
        att = tokenized['attention_mask'][0]

        word_ids = tokenized.word_ids()

        # check for a space before the word, if none is found either it is the first word or it should be removed
        delta = 0
        prev_word = None
        for i, w in enumerate(word_ids):
            if w is None or w == 0:
                continue
            char_span = tokenized.word_to_chars(w)
            if self.utterances[idx][char_span[0]-1] != ' ' and prev_word != w:
                # check if there is a space before the word
                delta += 1
            prev_word = w
            word_ids[i] = w - delta
        sentence_words = self.utterances[idx].split()

        # take only the first word piece of each word, keep the index of the first word piece
        words = set(word_ids)
        words.remove(None)
        mapping = torch.Tensor([word_ids.index(i) for i in set(words)])

        assert len(mapping) == len(sentence_words)

        slots = torch.Tensor(self.slot_ids[idx])
        intent = self.intent_ids[idx]
        sample = {'utterance': utt, 
                  'attention': att,
                  'mapping': mapping,
                  'sentence': sentence_words,
                  'slots': slots, 
                  'intent': intent}

        # breakpoint()
        return sample
    
    # Auxiliary methods
    
    def mapping_lab(self, data, mapper):
        '''
            Map labels to integers
        '''
        return [mapper[x] if x in mapper else mapper[self.unk] for x in data]
    
    def mapping_seq(self, data, mapper): # Map sequences to number
        '''
            Map sequences to integers
        '''
        res = []
        for seq in data:
            tmp_seq = []
            for x in seq.split():
                tmp_seq.append(mapper[x])
            res.append(tmp_seq)
        return res

from torch.utils.data import DataLoader

def collate_fn(data, lang, device="cuda"):
    '''
        Collate function to prepare the data for the model

        Args:
            data: list of samples
            lang: Lang object
            device: device to use

        Returns:
            new_item: dictionary with the utterances, intents, slots, slots_len, attention and mapping
    '''
    def merge(sequences, pad_token):
        '''
            merge from batch * sent_len to batch * max_len 

            Args:
                sequences: list of sequences
                pad_token: token to use for padding
        
            Returns:
                padded_seqs: tensor with the padded sequences
        '''
        # breakpoint()    
        lengths = [len(seq) for seq in sequences]
        max_len = 1 if max(lengths)==0 else max(lengths)
        # Pad token is zero in our case
        # So we create a matrix full of PAD_TOKEN (i.e. 0) with the shape 
        # batch_size X maximum length of a sequence
        padded_seqs = torch.LongTensor(len(sequences),max_len).fill_(pad_token)
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq # We copy each sequence into the matrix
        # print(padded_seqs)
        padded_seqs = padded_seqs.detach()  # We remove these tensors from the computational graph
        return padded_seqs, lengths
    
    # Sort data by seq lengths
    data.sort(key=lambda x: len(x['utterance']), reverse=True) 
    new_item = {}
    for key in data[0].keys():
        new_item[key] = [d[key] for d in data]
        
    # We just need one length for packed pad seq, since len(utt) == len(slots)
    src_utt, _ = merge(new_item['utterance'], lang.pad_token)
    y_slots, y_lengths = merge(new_item["slots"], lang.label_pad)
    att_masks, _ = merge(new_item["attention"], 0)
    mapping_padded, _ = merge(new_item["mapping"], 0)
    intent = torch.LongTensor(new_item["intent"])
    
    # We load the Tensor on our selected device
    src_utt = src_utt.to(device) 
    y_slots = y_slots.to(device)
    intent = intent.to(device)
    att_masks = att_masks.to(device)
    y_lengths = torch.LongTensor(y_lengths).to(device)
    
    new_item["utterances"] = src_utt
    new_item["intents"] = intent
    new_item["y_slots"] = y_slots
    new_item["slots_len"] = y_lengths
    new_item["attention"] = att_masks
    new_item["mapping"] = mapping_padded
    # breakpoint()
    return new_item

def load_data(path):
        '''
            input: path/to/data
            output: json 
        '''
        dataset = []
        with open(path) as f:
            dataset = json.loads(f.read())
        return dataset

def getDataLoaders(lang=None, batchsize=32, bert_model='bert-base-uncased', device="cuda"):
    '''
        Function to load the data and create the dataloaders

        Args:
            lang: Lang object
            batchsize: batch size
            bert_model: BERT model to use for tokenization
            device: device to use

        Returns:
            train_loader, dev_loader, test_loader, lang
    '''
    tmp_train_raw = load_data(os.path.join('..','dataset','ATIS','train.json'))
    test_raw = load_data(os.path.join('..','dataset','ATIS','test.json'))

    # First we get the 10% of the training set, then we compute the percentage of these examples 

    portion = 0.10

    intents = [x['intent'] for x in tmp_train_raw] # We stratify on intents
    count_y = Counter(intents)

    labels = []
    inputs = []
    mini_train = []

    for id_y, y in enumerate(intents):
        if count_y[y] > 1: # If some intents occurs only once, we put them in training
            inputs.append(tmp_train_raw[id_y])
            labels.append(y)
        else:
            mini_train.append(tmp_train_raw[id_y])
    # Random Stratify
    X_train, X_dev, _, _ = train_test_split(inputs, labels, test_size=portion, 
                                                        random_state=42, 
                                                        shuffle=True,
                                                        stratify=labels)
    X_train.extend(mini_train)
    train_raw = X_train
    dev_raw = X_dev

    corpus = train_raw + dev_raw + test_raw # We do not want unk labels, 
                                            # however this depends on the research purpose
    slots = set(sum([line['slots'].split() for line in corpus],[]))
    intents = set([line['intent'] for line in corpus])

    if lang is None:
        lang = Lang(intents, slots, bert_model=bert_model)

    # Create the datasets
    train_dataset = IntentsAndSlots(train_raw, lang)
    dev_dataset = IntentsAndSlots(dev_raw, lang)
    test_dataset = IntentsAndSlots(test_raw, lang)

    #send pad_token to the collate_fn
    train_loader = DataLoader(train_dataset, batch_size=batchsize, collate_fn=lambda x: collate_fn(x, lang=lang, device=device), shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=batchsize, collate_fn=lambda x: collate_fn(x, lang=lang, device=device))
    test_loader = DataLoader(test_dataset, batch_size=batchsize, collate_fn=lambda x: collate_fn(x, lang=lang, device=device))

    return train_loader, dev_loader, test_loader, lang