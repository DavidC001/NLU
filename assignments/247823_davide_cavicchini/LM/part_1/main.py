# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

# Import everything from functions.py file
from functions import train, init_weights
from utils import getDataLoaders
from model import LM_RNN, LM_LSTM
import torch
import os

batch_size = 256


if __name__ == "__main__":
    #Wrtite the code to load the datasets and to run your functions
    # Load the data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, dev_loader, test_loader, lang = getDataLoaders(batch_size=batch_size, device=device)

    run_exp = [1,1,1,1,1,1,1,1,1,1]
    
    #create models dir 
    if not os.path.exists("models"):
        os.makedirs("models")

    
    model = LM_RNN(emb_size=300, hidden_size=300, output_size=len(lang.word2id), n_layers=1).to(device)
    print(model)
    model = LM_LSTM(emb_size=300, hidden_size=300, output_size=len(lang.word2id), n_layers=1).to(device)
    print(model)

    # baseline model low LR
    if run_exp[0]:
        print("Baseline model low LR")
        model = LM_RNN(emb_size=300, hidden_size=300, output_size=len(lang.word2id), n_layers=1).to(device)
        model.apply(init_weights)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
        train(model, optimizer, "baseline_lowLR", lang, train_loader, dev_loader, test_loader, device=device)

    # baseline model
    if run_exp[1]:
        print("Baseline model")
        model = LM_RNN(emb_size=300, hidden_size=300, output_size=len(lang.word2id), n_layers=1).to(device)
        model.apply(init_weights)
        optimizer = torch.optim.SGD(model.parameters(), lr=1)
        train(model, optimizer, "baseline", lang, train_loader, dev_loader, test_loader, device=device)

    # higher LR
    if run_exp[2]:
        print("Baseline model with higher LR")
        model = LM_RNN(emb_size=300, hidden_size=300, output_size=len(lang.word2id), n_layers=1).to(device)
        model.apply(init_weights)
        optimizer = torch.optim.SGD(model.parameters(), lr=2)
        train(model, optimizer, "baseline_higherLR", lang, train_loader, dev_loader, test_loader, device=device)

    # LSTM low LR
    if run_exp[3]:
        print("LSTM model low LR")
        model = LM_LSTM(emb_size=300, hidden_size=300, output_size=len(lang.word2id), n_layers=1).to(device)
        model.apply(init_weights)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
        train(model, optimizer, "LSTM_lowLR", lang, train_loader, dev_loader, test_loader, device=device)

    # LSTM
    if run_exp[4]:
        print("LSTM model")
        model = LM_LSTM(emb_size=300, hidden_size=300, output_size=len(lang.word2id), n_layers=1).to(device)
        model.apply(init_weights)
        optimizer = torch.optim.SGD(model.parameters(), lr=1)
        train(model, optimizer, "LSTM", lang, train_loader, dev_loader, test_loader, device=device)

    # LSTM higher LR
    if run_exp[5]:
        print("LSTM model with higher LR")
        model = LM_LSTM(emb_size=300, hidden_size=300, output_size=len(lang.word2id), n_layers=1).to(device)
        model.apply(init_weights)
        optimizer = torch.optim.SGD(model.parameters(), lr=2)
        train(model, optimizer, "LSTM_higherLR", lang, train_loader, dev_loader, test_loader, device=device)

    # LSTM dropout low LR
    if run_exp[6]:
        print("LSTM model with dropout low LR")
        model = LM_LSTM(emb_size=300, hidden_size=300, output_size=len(lang.word2id), emb_dropout=0.25, out_dropout=0.25, n_layers=1).to(device)
        model.apply(init_weights)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
        train(model, optimizer, "LSTM_dropout_lowLR", lang, train_loader, dev_loader, test_loader, device=device)

    # LSTM dropout
    if run_exp[7]:
        print("LSTM model with dropout")
        model = LM_LSTM(emb_size=300, hidden_size=300, output_size=len(lang.word2id), emb_dropout=0.25, out_dropout=0.25, n_layers=1).to(device)
        model.apply(init_weights)
        optimizer = torch.optim.SGD(model.parameters(), lr=1)
        train(model, optimizer, "LSTM_dropout", lang, train_loader, dev_loader, test_loader, device=device)

    # LSTM dropout higher LR
    if run_exp[8]:
        print("LSTM model with dropout and higher LR")
        model = LM_LSTM(emb_size=300, hidden_size=300, output_size=len(lang.word2id), emb_dropout=0.25, out_dropout=0.25, n_layers=1).to(device)
        model.apply(init_weights)
        optimizer = torch.optim.SGD(model.parameters(), lr=2)
        train(model, optimizer, "LSTM_dropout_higherLR", lang, train_loader, dev_loader, test_loader, device=device)

    # LSTM AdamW
    if run_exp[9]:
        print("LSTM model with AdamW")
        model = LM_LSTM(emb_size=300, hidden_size=300, output_size=len(lang.word2id), emb_dropout=0.25, out_dropout=0.25, n_layers=1).to(device)
        model.apply(init_weights)
        optimizer = torch.optim.AdamW(model.parameters())
        train(model, optimizer, "LSTM_AdamW", lang, train_loader, dev_loader, test_loader, device=device)

