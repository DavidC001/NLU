# Add the class of your model only
# Here is where you define the architecture of your model using pytorch

from torch import nn
import math
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import copy
import numpy as np
from torch import optim
from utils import getDataLoaders
import os
from model import LM_LSTM_WT_VD

class NT_AvSGD(optim.SGD):
    """
    Implementation of Non-Monotonically Triggered Average Stochastic Gradient Descent (NT_AvSGD).

    Args:
        model (torch.nn.Module): The neural network model.
        dev_loader (torch.utils.data.DataLoader): The data loader for the validation set.
        criterion_eval (torch.nn.Module): The evaluation criterion.
        lr (float, optional): The learning rate (default: 1).
        L (int, optional): The number of iterations between validation checks (default: 165).
        n (int, optional): The number of previous validation checks to consider for non-monotonicity (default: 5).

    Attributes:
        temp (dict): A dictionary to store temporary parameter data during averaging.
        logs (list): A list to store the validation perplexity values.
        dev_loader (torch.utils.data.DataLoader): The data loader for the validation set.
        T (int): The iteration at which averaging is triggered.
        t (int): The number of cycles completed.
        k (int): The current iteration.
        L (int): The number of iterations between validation checks.
        n (int): The number of previous validation checks to consider for non-monotonicity.
        mu (int): The averaging parameter.
        model (torch.nn.Module): The neural network model.
        ax (dict): A dictionary to store the average of the parameters.
        criterion_eval (torch.nn.Module): The evaluation criterion.
    """

    def __init__(self, model, dev_loader, criterion_eval, lr=1, L=165, n=5):
        super(NT_AvSGD, self).__init__(model.parameters(), lr=lr)
        self.temp = {}
        self.logs = []
        self.dev_loader = dev_loader
        self.T = 0
        self.t = 0
        self.k = 0
        self.L = L
        self.n = n
        self.mu = 1
        self.model = model
        self.ax = {}
        self.criterion_eval = criterion_eval

    def step(self, closure=None):
        """
        Performs a single optimization step.
        """
        super(NT_AvSGD, self).step(closure)
        with torch.no_grad():
            # Calculate validation perplexity
            if self.k % self.L == 0 and self.T == 0:
                ppl_dev, _ = eval_loop(self.dev_loader, self.criterion_eval, self.model)
                self.model.train()
                if self.t > self.n and ppl_dev > min(self.logs[:self.t - self.n]):
                    self.T = self.k
                    print("Averaging started, at iteration", self.k, "after", self.t, "cycles")
                self.logs.append(ppl_dev)
                self.t += 1
        self.k += 1

        if self.T > 0:
            for prm in self.model.parameters():
                if prm not in self.ax:
                    self.ax[prm] = prm.data.clone()
                else:
                    self.ax[prm] = self.ax[prm] + (prm.data - self.ax[prm]) / self.mu
            self.mu += 1


    def average(self):
        """
        Performs parameter averaging.
        """
        if self.T == 0:
            # No need to average
            return
        with torch.no_grad():
            # Use ax computed in ASGD
            for prm in self.model.parameters():
                self.temp[prm] = prm.data.clone()
                prm.data = self.ax[prm].clone()

    def restore(self):
        """
        Restores the original parameter values.
        """
        if self.T == 0:
            # No need to restore
            return
        with torch.no_grad():
            for prm in self.model.parameters():
                prm.data = self.temp[prm].clone()


def train_loop(data, optimizer, criterion, model, clip=5):
    """
    Trains the model using the provided data, optimizer, criterion, and model.

    Args:
        data (iterable): The training data.
        optimizer (torch.optim.Optimizer): The optimizer used for updating the model's weights.
        criterion (torch.nn.modules.loss._Loss): The loss function used for computing the loss.
        model (torch.nn.Module): The model to be trained.
        clip (float, optional): The maximum gradient norm value to clip the gradients. Defaults to 5.

    Returns:
        float: The average loss per token over the training data.
    """
    model.train()
    loss_array = []
    number_of_tokens = []

    for sample in data:
        optimizer.zero_grad() # Zeroing the gradient
        output = model(sample['source'])
        loss = criterion(output, sample['target'])
        loss_array.append(loss.item() * sample["number_tokens"])
        number_of_tokens.append(sample["number_tokens"])
        loss.backward() # Compute the gradient, deleting the computational graph
        # clip the gradient to avoid explosioning gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step() # Update the weights

    return sum(loss_array)/sum(number_of_tokens)

def eval_loop(data, eval_criterion, model):
    """
    Evaluate the model on the given data using the specified evaluation criterion.

    Args:
        data (iterable): The data to evaluate the model on.
        eval_criterion (callable): The evaluation criterion to use.
        model: The model to evaluate.

    Returns:
        tuple: A tuple containing the perplexity (ppl) and the average loss (loss_to_return).
    """
    model.eval()
    loss_to_return = []
    loss_array = []
    number_of_tokens = []
    # softmax = nn.Softmax(dim=1) # Use Softmax if you need the actual probability
    with torch.no_grad(): # It used to avoid the creation of computational graph
        for sample in data:
            output = model(sample['source'])
            loss = eval_criterion(output, sample['target'])
            loss_array.append(loss.item())
            number_of_tokens.append(sample["number_tokens"])

    ppl = math.exp(sum(loss_array) / sum(number_of_tokens))
    loss_to_return = sum(loss_array) / sum(number_of_tokens)
    return ppl, loss_to_return

def init_weights(mat):
    """
    Initializes the weights of the given module using specific initialization techniques.

    Args:
        mat (nn.Module): The module for which to initialize the weights.
    """
    for m in mat.modules():
        if type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    for idx in range(4):
                        mul = param.shape[0]//4
                        torch.nn.init.xavier_uniform_(param[idx*mul:(idx+1)*mul])
                elif 'weight_hh' in name:
                    for idx in range(4):
                        mul = param.shape[0]//4
                        torch.nn.init.orthogonal_(param[idx*mul:(idx+1)*mul])
                elif 'bias' in name:
                    param.data.fill_(0)
        else:
            if type(m) in [nn.Linear]:
                torch.nn.init.uniform_(m.weight, -0.01, 0.01)
                if m.bias != None:
                    m.bias.data.fill_(0.01)

def train(model, optimizer, exp_name, 
          lang, train_loader, dev_loader, test_loader,
          clip=5, epochs=100, patience=3,
          tensorboard_folder='tensorboard', models_folder='models', device='cpu',
          model_folder='models'):
    """
    Trains the given model using the specified optimizer and data loaders.

    Args:
        model (nn.Module): The model to be trained.
        optimizer (torch.optim.Optimizer): The optimizer used for training.
        exp_name (str): The name of the experiment used for logging and saving the model.
        lang (Language): The language object containing ids of our vocabulary.
        train_loader (DataLoader): The data loader for training data.
        dev_loader (DataLoader): The data loader for development data.
        test_loader (DataLoader): The data loader for test data.
        clip (float, optional): The maximum gradient allowed. Defaults to 5.
        epochs (int, optional): The number of training epochs. Defaults to 100.
        patience (int, optional): The number of epochs to wait for improvement in validation loss before early stopping. Defaults to 3.
        tensorboard_folder (str, optional): The folder path for TensorBoard logs. Defaults to 'tensorboard'.
        models_folder (str, optional): The folder path for saving trained models. Defaults to 'models'.
        device (str, optional): The device to be used for training. Defaults to 'cpu'.
        model_folder (str, optional): The folder path for saving the trained model. Defaults to 'models'.
    """
    
    criterion_train = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"])
    criterion_eval = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"], reduction='sum')

    writer = SummaryWriter(tensorboard_folder+'/'+exp_name)

    best_ppl = math.inf
    pat = patience

    pbar = tqdm(range(1,epochs))
    #If the PPL is too high try to change the learning rate
    for epoch in pbar:
        loss = train_loop(train_loader, optimizer, criterion_train, model, clip)

        if epoch % 1 == 0:
            ppl_dev, loss_dev = eval_loop(dev_loader, criterion_eval, model)
            pbar.set_description("PPL: %f" % ppl_dev)

            writer.add_scalar('Loss/train', np.asarray(loss).mean(), epoch)
            writer.add_scalar('Loss/dev', np.asarray(loss_dev).mean(), epoch)
            writer.add_scalar('PPL/dev', ppl_dev, epoch)

            if  ppl_dev < best_ppl: # the lower, the better
                best_ppl = ppl_dev
                best_model = copy.deepcopy(model)
                patience = pat
            else:
                patience -= 1

            if patience <= 0: # Early stopping with patience
                break # Not nice but it keeps the code clean

    best_model.to(device)
    final_ppl,  _ = eval_loop(test_loader, criterion_eval, best_model)
    print('Test ppl: ', final_ppl)

    writer.add_scalar('PPL/test', final_ppl, 0)

    # Save the best model
    torch.save(best_model.state_dict(), os.path.join(model_folder, exp_name+".pt"))

    writer.close()

def runExps(run_exp = [1,1,1,1,1,1,1,1,1,1,1,1,1]):
    """
    Run the experiments for the different configurations of the models.
    """
    # Load the data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Define the batch size
    batch_size = 256
    train_loader, dev_loader, test_loader, lang = getDataLoaders(batch_size=batch_size, device=device)
    epochs = 200
    patience = 5

    #create models dir 
    model_path = "models"
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    # LSTM WT
    LSTM_WT = LM_LSTM_WT_VD(emb_size=300, hidden_size=300, output_size=len(lang.word2id), n_layers=1).to(device)
    LSTM_WT.apply(init_weights)
    exp_lr = [0.5, 1, 2]
    exp_name = ["LSTM_WT_lowLR", "LSTM_WT", "LSTM_WT_higherLR"]

    for i in range(len(exp_lr)):
        if run_exp[i]:
            print(f"Running {exp_name[i]} model with lr={exp_lr[i]}")
            model = copy.deepcopy(LSTM_WT).to(device)
            optimizer = torch.optim.SGD(model.parameters(), lr=exp_lr[i])
            train(model, optimizer, exp_name[i], lang, train_loader, dev_loader, test_loader, device=device, epochs=epochs, patience=patience, model_folder=model_path)
    
    # LSTM WT VD
    LSTM_WT_VD_base = LM_LSTM_WT_VD(emb_size=300, hidden_size=300, output_size=len(lang.word2id), emb_dropout=0.25, out_dropout=0.25, n_layers=1).to(device)
    LSTM_WT_VD_base.apply(init_weights)
    exp_lr = [0.5, 1, 2, 2.5, 3]
    exp_name = ["LSTM_WT_VD_lowLR", "LSTM_WT_VD", "LSTM_WT_VD_higherLR", "LSTM_WT_VD_even_higherLR", "LSTM_WT_VD_largeLR"]

    for i in range(len(exp_lr)):
        if run_exp[i+3]:
            print(f"Running {exp_name[i]} model with lr={exp_lr[i]}")
            model = copy.deepcopy(LSTM_WT_VD_base).to(device)
            optimizer = torch.optim.SGD(model.parameters(), lr=exp_lr[i])
            train(model, optimizer, exp_name[i], lang, train_loader, dev_loader, test_loader, device=device, epochs=epochs, patience=patience, model_folder=model_path)

    # LSTM WT VD NT_AvSGD low LR
    criterion = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"])
    exp_name = ["LSTM_NT_AvSGD_lowLR", "LSTM_NT_AvSGD", "LSTM_NT_AvSGD_higherLR", "LSTM_NT_AvSGD_even_higherLR", "LSTM_NT_AvSGD_largeLR"]

    for i in range(len(exp_lr)):
        if run_exp[i+8]:
            print(f"Running {exp_name[i]} model with lr={exp_lr[i]}")
            model = copy.deepcopy(LSTM_WT_VD_base).to(device)
            optimizer = NT_AvSGD(model, lr=exp_lr[i], dev_loader=dev_loader, criterion_eval=criterion)
            train(model, optimizer, exp_name[i], lang, train_loader, dev_loader, test_loader, device=device, epochs=epochs, patience=patience, model_folder=model_path)

def testModel(model, check, test, criterion, model_path):
    model.load_state_dict(torch.load(os.path.join(model_path, check)))
    ppl, _ = eval_loop(test, criterion, model)
    print(f"{check}: {ppl}")

def testModels():
    """
    Test the different configurations of the models on the test set.
    """
    # load the data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Define the batch size
    batch_size = 256
    _, _, test_loader, lang = getDataLoaders(batch_size=batch_size, device=device)
    criterion_eval = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"], reduction='sum')
    model_path = "models"

    # LSTM WT
    LSTM_WT = LM_LSTM_WT_VD(emb_size=300, hidden_size=300, output_size=len(lang.word2id), n_layers=1).to(device)
    model_names = ["LSTM_WT_lowLR", "LSTM_WT", "LSTM_WT_higherLR"]
    for check in model_names:
        testModel(LSTM_WT, check+".pt", test_loader, criterion_eval, model_path)
    
    # LSTM WT VD
    LSTM_WT_VD_base = LM_LSTM_WT_VD(emb_size=300, hidden_size=300, output_size=len(lang.word2id), emb_dropout=0.25, out_dropout=0.25, n_layers=1).to(device)
    model_names = ["LSTM_WT_VD_lowLR", "LSTM_WT_VD", "LSTM_WT_VD_higherLR", "LSTM_WT_VD_even_higherLR", "LSTM_WT_VD_largeLR", 
                     "LSTM_NT_AvSGD_lowLR", "LSTM_NT_AvSGD", "LSTM_NT_AvSGD_higherLR", "LSTM_NT_AvSGD_even_higherLR", "LSTM_NT_AvSGD_largeLR"]
    for check in model_names:
        testModel(LSTM_WT_VD_base, check+".pt", test_loader, criterion_eval, model_path)