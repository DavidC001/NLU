# Add the class of your model only
# Here is where you define the architecture of your model using pytorch

from torch import nn
import math
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import copy
import numpy as np
from torch.utils.data import DataLoader
from functools import partial

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
          tensorboard_folder='tensorboard', models_folder='models', device='cpu'):
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
    torch.save(best_model.state_dict(), models_folder+'/'+exp_name+'.pt')

    writer.close()
