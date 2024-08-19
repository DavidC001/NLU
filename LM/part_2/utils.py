# Add functions or classes used for data loading and preprocessing
import torch
import torch.utils.data as data
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from functools import partial


# Loading the corpus

def read_file(path, eos_token="<eos>"):
    """
    Read a file and return a list of sentences

    Args:
        path: str, path to the file
        eos_token: str, end of sentence token

    Returns:
        list of sentences
    """
    output = []
    with open(path, "r") as f:
        for line in f.readlines():
            output.append(line.strip() + " " + eos_token)
    return output

# Vocab with tokens to ids
def get_vocab(corpus, special_tokens=[]):
    """
    Returns a dictionary containing the vocabulary of the given corpus.

    Args:
        corpus (list): A list of sentences representing the corpus.
        special_tokens (list, optional): A list of special tokens to include in the vocabulary. Defaults to an empty list.

    Returns:
        dict: A dictionary where the keys are the unique words in the corpus (including special tokens) and the values are their corresponding indices.

    """
    output = {}
    i = 0
    for st in special_tokens:
        output[st] = i
        i += 1
    for sentence in corpus:
        for w in sentence.split():
            if w not in output:
                output[w] = i
                i += 1
    return output

# This class computes and stores our vocab
# Word to ids and ids to word
class Lang():
    """
    Class used to compute and store our vocab
    Word to ids and ids to word

    Args:
        corpus (list): A list of sentences representing the corpus.
        special_tokens (list, optional): A list of special tokens to include in the vocabulary. Defaults to an empty list.

    Attributes:
        word2id (dict): A dictionary mapping words to their corresponding IDs.
        id2word (dict): A dictionary mapping IDs to their corresponding words.

    Methods:
        get_vocab(corpus, special_tokens=[]): Returns a dictionary mapping words to their corresponding IDs.

    """
    def __init__(self, corpus, special_tokens=[]):
        self.word2id = self.get_vocab(corpus, special_tokens)
        self.id2word = {v:k for k, v in self.word2id.items()}

    def get_vocab(self, corpus, special_tokens=[]):
        """
        Returns a dictionary containing the vocabulary of the given corpus.

        Args:
            corpus (list): A list of sentences representing the corpus.
            special_tokens (list, optional): A list of special tokens to include in the vocabulary. Defaults to an empty list.

        Returns:
            dict: A dictionary where the keys are the unique words in the corpus (including special tokens) and the values are their corresponding indices.

        """
        output = {}
        i = 0
        for st in special_tokens:
            output[st] = i
            i += 1
        for sentence in corpus:
            for w in sentence.split():
                if w not in output:
                    output[w] = i
                    i += 1
        return output


class PennTreeBank (data.Dataset):
    """
    A PyTorch Dataset class for processing the Penn Treebank corpus.

    Args:
        corpus (list): List of sentences in the corpus.
        lang (Lang): Language object containing word-to-id mappings.

    Attributes:
        source (list): List of source sentences in the corpus.
        target (list): List of target sentences in the corpus.
        source_ids (list): List of source sentence IDs after mapping to word IDs.
        target_ids (list): List of target sentence IDs after mapping to word IDs.

    Methods:
        __len__(self): Returns the number of sentences in the corpus.
        __getitem__(self, idx): Returns the source and target sentences at the given index.
        mapping_seq(self, data, lang): Maps sequences of tokens to their corresponding IDs using the given language object.
    """

    # Mandatory methods are __init__, __len__ and __getitem__
    def __init__(self, corpus, lang):
        self.source = []
        self.target = []

        for sentence in corpus:
            self.source.append(sentence.split()[0:-1]) # We get from the first token till the second-last token
            self.target.append(sentence.split()[1:]) # We get from the second token till the last token
            # See example in section 6.2

        self.source_ids = self.mapping_seq(self.source, lang)
        self.target_ids = self.mapping_seq(self.target, lang)

    def __len__(self):
        return len(self.source)

    def __getitem__(self, idx):
        src= torch.LongTensor(self.source_ids[idx])
        trg = torch.LongTensor(self.target_ids[idx])
        sample = {'source': src, 'target': trg}
        return sample

    # Auxiliary methods

    def mapping_seq(self, data, lang): # Map sequences of tokens to corresponding computed in Lang class
        res = []
        for seq in data:
            tmp_seq = []
            for x in seq:
                if x in lang.word2id:
                    tmp_seq.append(lang.word2id[x])
                else:
                    print('OOV found!')
                    print('You have to deal with that') # PennTreeBank doesn't have OOV but "Trust is good, control is better!"
                    break
            res.append(tmp_seq)
        return res

def collate_fn(data, pad_token, device):
    """
    Collate function used to merge a list of samples into a batch.

    Args:
        data (list): A list of samples.
        pad_token (int): The ID of the padding token.
        device (str): The device to use (e.g., 'cpu' or 'cuda').
        
    Returns:
        dict: A dictionary containing the source and target sequences, as well as the number of tokens in the batch.
    """
    def merge(sequences):
        '''
        merge from batch * sent_len to batch * max_len
        '''
        lengths = [len(seq) for seq in sequences]
        max_len = 1 if max(lengths)==0 else max(lengths)
        # Pad token is zero in our case
        # So we create a matrix full of PAD_TOKEN (i.e. 0) with the shape
        # batch_size X maximum length of a sequence
        padded_seqs = torch.LongTensor(len(sequences),max_len).fill_(pad_token)
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq # We copy each sequence into the matrix
        padded_seqs = padded_seqs.detach()  # We remove these tensors from the computational graph
        return padded_seqs, lengths

    # Sort data by seq lengths

    data.sort(key=lambda x: len(x["source"]), reverse=True)
    new_item = {}
    for key in data[0].keys():
        new_item[key] = [d[key] for d in data]

    source, _ = merge(new_item["source"])
    target, lengths = merge(new_item["target"])

    new_item["source"] = source.to(device)
    new_item["target"] = target.to(device)
    new_item["number_tokens"] = sum(lengths)
    return new_item

def getDataLoaders(batch_size=256, device='cpu'):
    """
    Returns data loaders for training, development, and testing datasets.

    Args:
        batch_size (int): The batch size for the data loaders. Default is 256.
        device (str): The device to load the data on. Default is 'cpu'.

    Returns:
        tuple: A tuple containing the training data loader, development data loader,
               testing data loader, and the language object.

    """
    train_raw = read_file("../dataset/PennTreeBank/ptb.train.txt")
    dev_raw = read_file("../dataset/PennTreeBank/ptb.valid.txt")
    test_raw = read_file("../dataset/PennTreeBank/ptb.test.txt")

    # Vocab is computed only on training set
    # We add two special tokens end of sentence and padding
    vocab = get_vocab(train_raw, ["<pad>", "<eos>"])

    lang = Lang(train_raw, ["<pad>", "<eos>"])

    train_dataset = PennTreeBank(train_raw, lang)
    dev_dataset = PennTreeBank(dev_raw, lang)
    test_dataset = PennTreeBank(test_raw, lang)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"], device=device),  shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"], device=device))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"], device=device))

    return train_loader, dev_loader, test_loader, lang
