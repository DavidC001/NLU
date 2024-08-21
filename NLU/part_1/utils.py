from collections import Counter
import torch
import torch.utils.data as data
import json
import os
from sklearn.model_selection import train_test_split

# PAD token is used to pad the sequences to the same length
PAD_TOKEN = 0

class Lang():
    """
        Class to map words, intents and slots to integers
    """

    def __init__(self, words, intents, slots, cutoff=0):
        """
            Initialize the Lang class with the words, intents and slots

            Args:
                words : list of words
                intents : list of intents
                slots : list of slots
                cutoff : cutoff for the words
        """
        self.word2id = self.w2id(words, cutoff=cutoff, unk=True)
        self.slot2id = self.lab2id(slots)
        self.intent2id = self.lab2id(intents, pad=False)
        self.id2word = {v:k for k, v in self.word2id.items()}
        self.id2slot = {v:k for k, v in self.slot2id.items()}
        self.id2intent = {v:k for k, v in self.intent2id.items()}
        
    def w2id(self, elements, cutoff=None, unk=True):
        """
            Map words to integers

            Args:
                elements : list of words
                cutoff : cutoff for the words
                unk : whether to include the unknown token

            Returns:
                vocab : dictionary mapping words to integers
        """
        vocab = {'pad': PAD_TOKEN}
        if unk:
            vocab['unk'] = len(vocab)
        count = Counter(elements)
        for k, v in count.items():
            if v > cutoff:
                vocab[k] = len(vocab)
        return vocab
    
    def lab2id(self, elements, pad=True):
        """
            Map labels to integers

            Args:
                elements : list of labels
                pad : whether to include the padding token

            Returns:
                vocab : dictionary mapping labels to integers
        """
        vocab = {}
        if pad:
            vocab['pad'] = PAD_TOKEN
        for elem in elements:
                vocab[elem] = len(vocab)
        return vocab

class IntentsAndSlots (data.Dataset):
    """
        Dataset for Intents and Slots
    """
    
    def __init__(self, dataset, lang: Lang, unk='unk'):
        """
            Initialize the Dataset with the dataset and the language

            Args:
                dataset : list of examples
                lang : Lang object
                unk : unknown token
        """
        self.utterances = []
        self.intents = []
        self.slots = []
        self.unk = unk
        
        for x in dataset:
            self.utterances.append(x['utterance'])
            self.slots.append(x['slots'])
            self.intents.append(x['intent'])

        # Map the sequences to integers using the Lang object
        self.utt_ids = self.mapping_seq(self.utterances, lang.word2id)
        self.slot_ids = self.mapping_seq(self.slots, lang.slot2id)
        self.intent_ids = self.mapping_lab(self.intents, lang.intent2id)

        # breakpoint()

    def __len__(self):
        """
            Return the length of the dataset

            Returns:
                length of the dataset
        """
        return len(self.utterances)

    def __getitem__(self, idx):
        """
            Get an example from the dataset

            Args:
                idx: index of the example

            Returns:
                sample: dictionary with the utterance, slots and intent with the following keys:
                    - utterance: tensor with the utterance tokens
                    - slots: tensor with the slots
                    - intent: tensor with the intent
        """
        utt = torch.Tensor(self.utt_ids[idx])
        slots = torch.Tensor(self.slot_ids[idx])
        intent = self.intent_ids[idx]
        
        sample = {'utterance': utt, 'slots': slots, 'intent': intent}
        return sample
    
    # Auxiliary methods
    
    def mapping_lab(self, data, mapper):
        """
            Map labels to integers
        """
        return [mapper[x] if x in mapper else mapper[self.unk] for x in data]
    
    def mapping_seq(self, data, mapper):
        """
            Map sequences to integers
        """
        res = []
        for seq in data:
            tmp_seq = []
            for x in seq.split():
                if x in mapper:
                    tmp_seq.append(mapper[x])
                else:
                    tmp_seq.append(mapper[self.unk])
            res.append(tmp_seq)
        return res

from torch.utils.data import DataLoader

def collate_fn(data, device="cuda"):
    '''
        Collate function to prepare the data for the model

        Args:
            data: list of examples
            device: device to use

        Returns:
            new_item: dictionary with the utterances, intents, slots and slot lengths padded and on the device
    '''
    def merge(sequences):
        '''
            merge from batch * sent_len to batch * max_len 
        '''
        lengths = [len(seq) for seq in sequences]
        max_len = 1 if max(lengths)==0 else max(lengths)

        # Pad token is zero in our case
        # So we create a matrix full of PAD_TOKEN (i.e. 0) with the shape 
        # batch_size X maximum length of a sequence
        padded_seqs = torch.LongTensor(len(sequences),max_len).fill_(PAD_TOKEN)
        
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq # We copy each sequence into the matrix
        
        padded_seqs = padded_seqs.detach()  # We remove these tensors from the computational graph

        return padded_seqs, lengths
    
    # Sort data by seq lengths
    data.sort(key=lambda x: len(x['utterance']), reverse=True) 

    # Construct the new item
    new_item = {}
    for key in data[0].keys():
        new_item[key] = [d[key] for d in data]
        
    # We just need one length for packed pad seq, since len(utt) == len(slots)
    src_utt, _ = merge(new_item['utterance'])
    y_slots, y_lengths = merge(new_item["slots"])
    intent = torch.LongTensor(new_item["intent"])
    
    # We load the Tensors on our selected device
    src_utt = src_utt.to(device) 
    y_slots = y_slots.to(device)
    intent = intent.to(device)
    y_lengths = torch.LongTensor(y_lengths).to(device)
    
    # We update the new_item dictionary and return it
    new_item["utterances"] = src_utt
    new_item["intents"] = intent
    new_item["y_slots"] = y_slots
    new_item["slots_len"] = y_lengths
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

def getDataLoaders(lang=None, batchsize=32, device="cuda"):
    '''
        Function to load the data and create the data loaders

        Args:
            lang: Lang object
            batchsize: batch size
            device: device to use

        Returns:
            out: train_loader, dev_loader, test_loader, lang
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

    words = sum([x['utterance'].split() for x in train_raw], []) # No set() since we want to compute the cutoff
    corpus = train_raw + dev_raw + test_raw # We do not want unk labels, however this depends on the research purpose
    slots = set(sum([line['slots'].split() for line in corpus],[]))
    intents = set([line['intent'] for line in corpus])

    # If we do not have a lang object, we create a new one with the data we have
    if lang is None:
        lang = Lang(words, intents, slots, cutoff=0)

    # Create the datasets
    train_dataset = IntentsAndSlots(train_raw, lang)
    dev_dataset = IntentsAndSlots(dev_raw, lang)
    test_dataset = IntentsAndSlots(test_raw, lang)

    # Create the data loaders
    train_loader = DataLoader(train_dataset, batch_size=batchsize, collate_fn=lambda x: collate_fn(x, device=device), shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=batchsize, collate_fn=lambda x: collate_fn(x, device=device))
    test_loader = DataLoader(test_dataset, batch_size=batchsize, collate_fn=lambda x: collate_fn(x, device=device))

    return train_loader, dev_loader, test_loader, lang