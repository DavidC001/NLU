# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

# Import everything from functions.py file
from functions import *
import torch
from utils import getDataLoaders
from model import ModelIAS
import numpy as np
import argparse

#silence warnings
import warnings
warnings.filterwarnings("ignore")

torch.manual_seed(0)
np.random.seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batchsize = 128 # large is bad, but small requires smaller learning rate

def train():
    _, _, _, lang = getDataLoaders(batchsize=batchsize)

    out_slot = len(lang.slot2id)
    out_int = len(lang.intent2id)

    tests_base = {
        'device': device,

        'bert_model': 'bert-base-uncased',
        'classification_layers_slots': [],
        'classification_layers_intents': [],
        'dropoutBertEmb': 0,
        
        'lr': 0.0005,
        'n_epochs': 200,
        'runs': 5,
        'clip': 5,
        'patience': 6,
        'batchsize': batchsize,
        
        'test_name': 'base',
    }

    tests = [
        {
            'bert_model': 'bert-base-uncased',
            'test_name': "BERT_base_uncased",
        },
        {
            'bert_model': 'bert-base-cased',
            'test_name': "BERT_base_cased",
        },
        {
            'bert_model': 'bert-large-uncased',
            'test_name': "BERT_large_uncased",
        },
        {
            'bert_model': 'bert-large-cased',
            'test_name': "BERT_large_cased",
        }
    ]

    for test in tests:
        # modify the base test with the new parameters
        test = {**tests_base, **test}
        # breakpoint()
        runTest(**test)

def test():
    for file in os.listdir('models'):
        #load object        }
        saved_object = torch.load(os.path.join('models', file))
        lang = saved_object['lang']
        model_params = saved_object['model_params']
        model = ModelIAS(**model_params).to(device)
        model.load_state_dict(saved_object['model'])
        
        _, _, test_loader, lang = getDataLoaders(batchsize=batchsize, lang=lang)
        
        results_test, intent_test, _ = eval_loop(test_loader, model, lang)
        print(file)
        print('Intent Acc', intent_test['accuracy'])
        print('Slot F1', results_test['total']['f'])

        # breakpoint()

if __name__ == "__main__":

    # define command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--Train', type=bool, default=False)
    parser.add_argument('--Test', type=bool, default=True)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.Train:
        train()

    if args.Test:
        test()
