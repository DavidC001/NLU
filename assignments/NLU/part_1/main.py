# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

# Import everything from functions.py file
from functions import *
import torch
from utils import PAD_TOKEN, getDataLoaders
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
    train_loader, dev_loader, test_loader, lang = getDataLoaders(batchsize=batchsize)

    out_slot = len(lang.slot2id)
    out_int = len(lang.intent2id)
    vocab_len = len(lang.word2id)

    tests_base = {
        'device': device,
        'lang': lang,
        'out_slot': out_slot,
        'out_int': out_int,
        'vocab_len': vocab_len,
        'pad_index': PAD_TOKEN,

        'train_loader': train_loader,
        'dev_loader': dev_loader,
        'test_loader': test_loader,

        'hid_size': 256,
        'emb_size': 512,
        'n_layer': 1,
        
        'lr': 0.0005,
        'n_epochs': 200,
        'runs': 10,
        'clip': 5,
        'patience': 10,
        
        'dropoutEmb': 0,
        'dropoutOut': 0,
        'bidirectional': False,
        'combine': 'concat',
        'layerNorm': False,
        
        'test_name': 'base',
    }

    tests = [
        {
            'test_name': "baseline",
        },
        {
            'test_name': "bidirectional_concat",
            'bidirectional': True
        },
        {   
            'test_name': "bidirectional_sum",
            'bidirectional': True,
            'combine': 'sum'
        },
        {
            'test_name': "bidirectional_gated",
            'bidirectional': True,
            'combine': 'gated'
        },
        {
            'test_name': "dropout_concat",
            'dropoutEmb': 0.5,
            'dropoutOut': 0.5,
            'bidirectional': True,
            'combine': 'concat'
        },
        {
            'test_name': "dropout_sum",
            'dropoutEmb': 0.5,
            'dropoutOut': 0.5,
            'bidirectional': True,
            'combine': 'sum'
        },
        {
            'test_name': "dropout_gated",
            'dropoutEmb': 0.5,
            'dropoutOut': 0.5,
            'bidirectional': True,
            'combine': 'gated'
        },
        {
            'test_name': "dropout_gated_LN",
            'dropoutEmb': 0.5,
            'dropoutOut': 0.5,
            'bidirectional': True,
            'combine': 'gated',
            'layerNorm': True
        },
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
