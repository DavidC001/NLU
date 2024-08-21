# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

# Import everything from functions.py file
from functions import *
import torch
from utils import getDataLoaders
from model import ModelSA
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

    tests_base = {
        'device': device,

        'bert_model': 'bert-base-uncased',
        'classification_layers': [],
        'dropoutBertEmb': 0,
        
        'lr': 0.0001,
        'n_epochs': 200,
        'runs': 5,
        'clip': 5,
        'patience': 5,
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
            #roberta
            'bert_model': 'roberta-base',
            'test_name': "RoBERTa_base",
        }
    ]

    for test in tests:
        # modify the base test with the new parameters
        test = {**tests_base, **test}
        # breakpoint()
        runTest(**test)

def test():
    for file in os.listdir('models'):
        print(file)
        #load object        }
        saved_object = torch.load(os.path.join('models', file))
        lang = saved_object['lang']
        model_params = saved_object['model_params']
        model = ModelSA(**model_params).to(device)
        model.load_state_dict(saved_object['model'])

        # print saved scores and stds
        print("\tResults:")
        print(f"\t\tMacro F1: {saved_object['results']['macro_f1']} +/- {saved_object['results']['macro_f1_std']}")
        print(f"\t\tMicro F1: {saved_object['results']['micro_f1']} +/- {saved_object['results']['micro_f1_std']}")
        print(f"\t\tPrecision: {saved_object['results']['precision']} +/- {saved_object['results']['precision_std']}")
        print(f"\t\tRecall: {saved_object['results']['recall']} +/- {saved_object['results']['recall_std']}")
        
        _, _, test_loader, lang = getDataLoaders(batchsize=batchsize, lang=lang, bert_model=model_params['bert_model'])

        # evaluate the model
        results_test, _ = eval_loop(test_loader, model, lang)
        print('\tResults on test set for best saved model:')
        print('\t\tMacro F1', results_test['macro f1'])
        print('\t\tMicro F1', results_test['micro f1'])
        print('\t\tPrecision', results_test['micro p'])
        print('\t\tRecall', results_test['micro r'])

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
