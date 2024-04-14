# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

# Import everything from functions.py file
from functions import train, NT_AvSGD, init_weights
from utils import getDataLoaders
from model import LM_LSTM_WT_VD
import torch
from torch import nn
import os

batch_size = 256


if __name__ == "__main__":
    #Wrtite the code to load the datasets and to run your functions
    # Load the data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, dev_loader, test_loader, lang = getDataLoaders(batch_size=batch_size, device=device)

    run_exp = [0,0,0,0,0,0,1,1,1,1]

    #create models dir 
    if not os.path.exists("models"):
        os.makedirs("models")

    # LSTM WT low LR
    if run_exp[0]:
        print("LSTM WT model low LR")
        model = LM_LSTM_WT_VD(emb_size=300, hidden_size=300, output_size=len(lang.word2id), n_layers=1).to(device)
        model.apply(init_weights)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
        train(model, optimizer, "LSTM_WT_lowLR", lang, train_loader, dev_loader, test_loader, device=device)

    # LSTM WT
    if run_exp[1]:
        print("LSTM WT model")
        model = LM_LSTM_WT_VD(emb_size=300, hidden_size=300, output_size=len(lang.word2id), n_layers=1).to(device)
        model.apply(init_weights)
        optimizer = torch.optim.SGD(model.parameters(), lr=1)
        train(model, optimizer, "LSTM_WT", lang, train_loader, dev_loader, test_loader, device=device)

    # LSTM WT higher LR
    if run_exp[2]:
        print("LSTM WT model with higher LR")
        model = LM_LSTM_WT_VD(emb_size=300, hidden_size=300, output_size=len(lang.word2id), n_layers=1).to(device)
        model.apply(init_weights)
        optimizer = torch.optim.SGD(model.parameters(), lr=2)
        train(model, optimizer, "LSTM_WT_higherLR", lang, train_loader, dev_loader, test_loader, device=device)

    # LSTM WT VD low LR
    if run_exp[3]:
        print("LSTM WT model with var dropout low LR")
        model = LM_LSTM_WT_VD(emb_size=300, hidden_size=300, output_size=len(lang.word2id), emb_dropout=0.25, out_dropout=0.25, n_layers=1).to(device)
        model.apply(init_weights)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
        train(model, optimizer, "LSTM_WT_VD_lowLR", lang, train_loader, dev_loader, test_loader, device=device)

    # LSTM WT VD
    if run_exp[4]:
        print("LSTM WT model with var dropout")
        model = LM_LSTM_WT_VD(emb_size=300, hidden_size=300, output_size=len(lang.word2id), emb_dropout=0.25, out_dropout=0.25, n_layers=1).to(device)
        model.apply(init_weights)
        optimizer = torch.optim.SGD(model.parameters(), lr=1)
        train(model, optimizer, "LSTM_WT_VD", lang, train_loader, dev_loader, test_loader, device=device)

    # LSTM WT VD higher LR
    if run_exp[5]:
        print("LSTM WT model with var dropout and higher LR")
        model = LM_LSTM_WT_VD(emb_size=300, hidden_size=300, output_size=len(lang.word2id), emb_dropout=0.25, out_dropout=0.25, n_layers=1).to(device)
        model.apply(init_weights)
        optimizer = torch.optim.SGD(model.parameters(), lr=2)
        train(model, optimizer, "LSTM_WT_VD_higherLR", lang, train_loader, dev_loader, test_loader, device=device)

    # LSTM WT VD NT_AvSGD low LR
    criterion = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"])
    if run_exp[6]:
        print("LSTM WT model with NT_AvSGD low LR")
        model = LM_LSTM_WT_VD(emb_size=300, hidden_size=300, output_size=len(lang.word2id), emb_dropout=0.25, out_dropout=0.25, n_layers=1).to(device)
        model.apply(init_weights)
        optimizer = NT_AvSGD(model, lr=0.5, n=5, dev_loader=dev_loader, criterion=criterion)
        train(model, optimizer, "LSTM_NT_AvSGD_lowLR", lang, train_loader, dev_loader, test_loader, device=device, patience=10)

    # LSTM NT_AvSGD
    if run_exp[7]:
        print("LSTM model with NT_AvSGD")
        model = LM_LSTM_WT_VD(emb_size=300, hidden_size=300, output_size=len(lang.word2id), emb_dropout=0.25, out_dropout=0.25, n_layers=1).to(device)
        model.apply(init_weights)
        optimizer = NT_AvSGD(model, lr=1, n=5, dev_loader=dev_loader, criterion=criterion)
        train(model, optimizer, "LSTM_NT_AvSGD", lang, train_loader, dev_loader, test_loader, device=device, patience=10)


    # LSTM NT_AvSGD higher LR
    if run_exp[8]:
        print("LSTM model with NT_AvSGD and higher LR")
        model = LM_LSTM_WT_VD(emb_size=300, hidden_size=300, output_size=len(lang.word2id), emb_dropout=0.25, out_dropout=0.25, n_layers=1).to(device)
        model.apply(init_weights)
        optimizer = NT_AvSGD(model, lr=2, n=5, dev_loader=dev_loader, criterion=criterion)
        train(model, optimizer, "LSTM_NT_AvSGD_higherLR", lang, train_loader, dev_loader, test_loader, device=device, patience=10)
    
    # LSTM NT_AvSGD even higher LR
    if run_exp[9]:
        print("LSTM model with NT_AvSGD and even higher LR")
        model = LM_LSTM_WT_VD(emb_size=300, hidden_size=300, output_size=len(lang.word2id), emb_dropout=0.25, out_dropout=0.25, n_layers=1).to(device)
        model.apply(init_weights)
        optimizer = NT_AvSGD(model, lr=2.5, n=5, dev_loader=dev_loader, criterion=criterion)
        train(model, optimizer, "LSTM_NT_AvSGD_even_higherLR", lang, train_loader, dev_loader, test_loader, device=device, patience=10)
