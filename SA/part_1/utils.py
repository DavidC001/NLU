import torch
import torch.utils.data as data
import os
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from nltk import word_tokenize

class Lang():
    def __init__(self, sentiments, bert_model):
        # self.word2id = self.w2id(words, cutoff=cutoff, unk=True)
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model)
        self.pad_token = self.tokenizer.pad_token_id
        self.label_pad = 0
        self.sent2id = self.lab2id(sentiments)
        # self.id2word = {v:k for k, v in self.word2id.items()}
        self.id2sent = {v:k for k, v in self.sent2id.items()}
    
    def lab2id(self, elements, pad=True):
        vocab = {}
        if pad:
            vocab['pad'] = 0
        for elem in elements:
                vocab[elem] = len(vocab)
        return vocab

class SADataset (data.Dataset):
    # Mandatory methods are __init__, __len__ and __getitem__
    def __init__(self, dataset, lang, unk='unk'):
        self.utterances = []
        self.sentiments = []
        self.words = []
        self.tokenizer = lang.tokenizer
        self.unk = unk
        
        for x in dataset:
            self.utterances.append(x['utterance'])
            self.sentiments.append(x['sentiment'])
            self.words.append(x['words'])

        self.sent_ids = self.mapping_seq(self.sentiments, lang.sent2id)

    def __len__(self):
        return len(self.utterances)

    def __getitem__(self, idx):
        #join text from list of words
        text = ' '.join(self.words[idx])
        tokenized = self.tokenizer(text, return_tensors='pt')
        utt = tokenized['input_ids'][0]
        att = tokenized['attention_mask'][0]

        word_ids = tokenized.word_ids()

        # breakpoint() 
        # check for a space before the word, if none is found either it is the first word or it should be removed
        delta = 0
        prev_word = None
        for i, w in enumerate(word_ids):
            if w is None or w == 0:
                continue
            char_span = tokenized.word_to_chars(w)
            if text[char_span[0]-1] != ' ' and prev_word != w:
                # check if there is a space before the word and the word before has it
                delta += 1
            prev_word = w
            word_ids[i] = w - delta

        # take only the first word piece of each word, keep the index of the first word piece
        words = set(word_ids)
        words.remove(None)
        mapping = torch.Tensor([word_ids.index(i) for i in set(words)])


        sentiment = torch.Tensor(self.sent_ids[idx])

        assert len(self.words[idx]) == len(mapping)

        sample = {'utterance': utt, 
                  'attention': att,
                  'mapping': mapping,
                  'sentence': self.words[idx],
                  'sentiment': sentiment, 
                  }

        # breakpoint()
        return sample
    
    # Auxiliary methods
    
    def mapping_seq(self, data, mapper): # Map sequences to number
        res = []
        for seq in data:
            tmp_seq = []
            for x in seq:
                if x in mapper:
                    tmp_seq.append(mapper[x])
                else:
                    tmp_seq.append(mapper[self.unk])
            res.append(tmp_seq)
        return res

from torch.utils.data import DataLoader

def collate_fn(data, lang, device="cuda"):
    def merge(sequences, pad_token):
        '''
        merge from batch * sent_len to batch * max_len 
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
    y_sents, y_lengths = merge(new_item["sentiment"], lang.label_pad)
    att_masks, _ = merge(new_item["attention"], 0)
    # breakpoint()
    mapping_padded, _ = merge(new_item["mapping"], 0)
    
    src_utt = src_utt.to(device) # We load the Tensor on our selected device
    y_sents = y_sents.to(device)
    att_masks = att_masks.to(device)
    y_lengths = torch.LongTensor(y_lengths).to(device)
    
    new_item["utterances"] = src_utt
    new_item["y_sents"] = y_sents
    new_item["sent_len"] = y_lengths
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
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                sample = {}
                line = line.split("####")
                sample['utterance'] = line[0]
                sample['sentiment'] = [s.split("=")[-1] for s in line[1].split()]
                sample['words'] = [s.split("=")[0] if s[0]!="=" else s[1:] for s in line[1].split()]
                dataset.append(sample)
        return dataset

def getDataLoaders(lang=None, batchsize=32, bert_model='bert-base-uncased', device="cuda"):
    tmp_train_raw = load_data(os.path.join('..','dataset','laptop14_train.txt'))
    test_raw = load_data(os.path.join('..','dataset','laptop14_test.txt'))

    # First we get the 10% of the training set, then we compute the percentage of these examples 

    portion = 0.10

    # count the number phrases a sentiment appears in (probably unecessary, wanted to keep the code similar to the NLU part)
    sentiments = {}
    for x in tmp_train_raw:
        for s in set(x['sentiment']):
            if s not in sentiments:
                sentiments[s] = 0
            sentiments[s] += 1

    labels = []
    inputs = []
    mini_train = []

    for id_y, y in enumerate(tmp_train_raw):
        if not any([sentiments[s] == 1 for s in y['sentiment']]):
            inputs.append(tmp_train_raw[id_y])
            labels.append(y['sentiment'])
        else:
            mini_train.append(tmp_train_raw[id_y])
    # breakpoint()
    
    # Random Stratify
    X_train, X_dev, _, _ = train_test_split(inputs, labels, test_size=portion, 
                                                        random_state=42, 
                                                        shuffle=True)
    X_train.extend(mini_train)
    train_raw = X_train
    dev_raw = X_dev

    corpus = train_raw + dev_raw + test_raw # We do not want unk labels, 
                                            # however this depends on the research purpose
    sentiments = set([s for line in corpus for s in line['sentiment']])
    # breakpoint()

    if lang is None:
        lang = Lang(sentiments, bert_model=bert_model)

    train_dataset = SADataset(train_raw, lang)
    dev_dataset = SADataset(dev_raw, lang)
    test_dataset = SADataset(test_raw, lang)

    #send pad_token to the collate_fn
    train_loader = DataLoader(train_dataset, batch_size=batchsize, collate_fn=lambda x: collate_fn(x, lang=lang, device=device), shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=batchsize, collate_fn=lambda x: collate_fn(x, lang=lang, device=device))
    test_loader = DataLoader(test_dataset, batch_size=batchsize, collate_fn=lambda x: collate_fn(x, lang=lang, device=device))

    return train_loader, dev_loader, test_loader, lang