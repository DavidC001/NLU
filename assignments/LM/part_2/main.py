# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

# Import everything from functions.py file
from functions import train, NT_AvSGD, init_weights
from utils import getDataLoaders
from model import LM_LSTM_WT_VD
import torch
from torch import nn
import os
import copy

# Define the batch size
batch_size = 256


if __name__ == "__main__":
    #Wrtite the code to load the datasets and to run your functions
    # Load the data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, dev_loader, test_loader, lang = getDataLoaders(batch_size=batch_size, device=device)

    # Define the experiments to run
    run_exp = [1,1,1,1,1,1,1,1,1,1,1,1,1]

    #create models dir 
    if not os.path.exists("models"):
        os.makedirs("models")

    LSTM_WT = LM_LSTM_WT_VD(emb_size=300, hidden_size=300, output_size=len(lang.word2id), n_layers=1).to(device)
    LSTM_WT.apply(init_weights)

    # LSTM WT low LR
    if run_exp[0]:
        print("LSTM WT model low LR")
        model = copy.deepcopy(LSTM_WT).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
        train(model, optimizer, "LSTM_WT_lowLR", lang, train_loader, dev_loader, test_loader, device=device, epochs=200, patience=5)

    # LSTM WT
    if run_exp[1]:
        print("LSTM WT model")
        model = copy.deepcopy(LSTM_WT).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=1)
        train(model, optimizer, "LSTM_WT", lang, train_loader, dev_loader, test_loader, device=device, epochs=200, patience=5)

    # LSTM WT higher LR
    if run_exp[2]:
        print("LSTM WT model with higher LR")
        model = copy.deepcopy(LSTM_WT).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=2)
        train(model, optimizer, "LSTM_WT_higherLR", lang, train_loader, dev_loader, test_loader, device=device, epochs=200, patience=5)

    LSTM_WT_VD_base = LM_LSTM_WT_VD(emb_size=300, hidden_size=300, output_size=len(lang.word2id), emb_dropout=0.25, out_dropout=0.25, n_layers=1).to(device)
    LSTM_WT_VD_base.apply(init_weights)

    # LSTM WT VD low LR
    if run_exp[3]:
        print("LSTM WT model with var dropout low LR")
        model = copy.deepcopy(LSTM_WT_VD_base).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
        train(model, optimizer, "LSTM_WT_VD_lowLR", lang, train_loader, dev_loader, test_loader, device=device, epochs=200, patience=5)

    # LSTM WT VD
    if run_exp[4]:
        print("LSTM WT model with var dropout")
        model = copy.deepcopy(LSTM_WT_VD_base).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=1)
        train(model, optimizer, "LSTM_WT_VD", lang, train_loader, dev_loader, test_loader, device=device, epochs=200, patience=5)

    # LSTM WT VD higher LR
    if run_exp[5]:
        print("LSTM WT model with var dropout and higher LR")
        model = copy.deepcopy(LSTM_WT_VD_base).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=2)
        train(model, optimizer, "LSTM_WT_VD_higherLR", lang, train_loader, dev_loader, test_loader, device=device, epochs=200, patience=5)

    # LSTM WT VD even higher LR
    if run_exp[6]:
        print("LSTM WT model with var dropout and even higher LR")
        model = copy.deepcopy(LSTM_WT_VD_base).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=2.5)
        train(model, optimizer, "LSTM_WT_VD_even_higherLR", lang, train_loader, dev_loader, test_loader, device=device, epochs=200, patience=5)

    
    # LSTM WT VD large LR
    if run_exp[7]:
        print("LSTM WT model with var dropout and large LR")
        model = copy.deepcopy(LSTM_WT_VD_base).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=3)
        train(model, optimizer, "LSTM_WT_VD_largeLR", lang, train_loader, dev_loader, test_loader, device=device, epochs=200, patience=5)

    # LSTM WT VD NT_AvSGD low LR
    criterion = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"])
    if run_exp[8]:
        print("LSTM WT model with NT_AvSGD low LR")
        model = copy.deepcopy(LSTM_WT_VD_base).to(device)
        optimizer = NT_AvSGD(model, lr=0.5, dev_loader=dev_loader, criterion_eval=criterion)
        train(model, optimizer, "LSTM_NT_AvSGD_lowLR", lang, train_loader, dev_loader, test_loader, device=device, epochs=200, patience=5)

    # LSTM NT_AvSGD
    if run_exp[9]:
        print("LSTM model with NT_AvSGD")
        model = copy.deepcopy(LSTM_WT_VD_base).to(device)
        optimizer = NT_AvSGD(model, lr=1, dev_loader=dev_loader, criterion_eval=criterion)
        train(model, optimizer, "LSTM_NT_AvSGD", lang, train_loader, dev_loader, test_loader, device=device, epochs=200, patience=5)


    # LSTM NT_AvSGD higher LR
    if run_exp[10]:
        print("LSTM model with NT_AvSGD and higher LR")
        model = copy.deepcopy(LSTM_WT_VD_base).to(device)
        optimizer = NT_AvSGD(model, lr=2, dev_loader=dev_loader, criterion_eval=criterion)
        train(model, optimizer, "LSTM_NT_AvSGD_higherLR", lang, train_loader, dev_loader, test_loader, device=device, epochs=200, patience=5)
    
    # LSTM NT_AvSGD even higher LR
    if run_exp[11]:
        print("LSTM model with NT_AvSGD and even higher LR")
        model = copy.deepcopy(LSTM_WT_VD_base).to(device)
        optimizer = NT_AvSGD(model, lr=2.5, dev_loader=dev_loader, criterion_eval=criterion)
        train(model, optimizer, "LSTM_NT_AvSGD_even_higherLR", lang, train_loader, dev_loader, test_loader, device=device, epochs=200, patience=5)

    # LSTM NT_AvSGD large LR
    if run_exp[12]:
        print("LSTM model with NT_AvSGD and large LR")
        model = copy.deepcopy(LSTM_WT_VD_base).to(device)
        optimizer = NT_AvSGD(model, lr=3, dev_loader=dev_loader, criterion_eval=criterion)
        train(model, optimizer, "LSTM_NT_AvSGD_largeLR", lang, train_loader, dev_loader, test_loader, device=device, epochs=200, patience=5)
