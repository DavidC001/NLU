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
from torch import optim

class NT_AvSGD(optim.ASGD):
    def __init__(self, model, dev_loader, criterion_eval, lr=1, L=165, n=5):
        super(NT_AvSGD, self).__init__(model.parameters(), lr=lr, t0=math.inf)
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
        self.criterion_eval = criterion_eval

    def step(self, closure=None):
        super(NT_AvSGD, self).step(closure)
        with torch.no_grad():
          #calculate validation PPL
          if self.k % self.L == 0 and self.T==0:
              ppl_dev, _ = eval_loop(self.dev_loader, self.criterion_eval, self.model)
              self.model.train()
              if self.t>self.n and ppl_dev > min(self.logs[:self.t-self.n]):
                self.T = self.k
                # set t0 of ASGD to 0
                for group in self.param_groups:
                    group['t0'] = 0
                print("averaging started, at iteration", self.k, " after ", self.t, " cycles")
              self.logs.append(ppl_dev)
              self.t += 1
          self.k += 1

    def average(self):
        if self.T == 0:
            #print("No need to average")
            return
        with torch.no_grad():
            # use ax computed in ASGD
            for prm in self.model.parameters():
                self.temp[prm] = prm.data.clone()
                prm.data = self.state[prm]['ax'].clone()

    def restore(self):
        if self.T == 0:
            #print("No need to restore")
            return
        with torch.no_grad():
            for prm in self.model.parameters():
                prm.data = self.temp[prm].clone()


def train_loop(data, optimizer, criterion, model, clip=5):
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

            
            # if t0 in optimizer is set to 0, the model is averaged
            if 't0' in optimizer.param_groups[0]:
                    optimizer.average()
                    ppl_dev, loss_dev = eval_loop(dev_loader, criterion_eval, model)
                    optimizer.restore()
                    writer.add_scalar('PPL/dev_avg', ppl_dev, epoch)

            if  ppl_dev < best_ppl: # the lower, the better (note we are looking at the averaged in case of ASGD)
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

    if 't0' in optimizer.param_groups[0]:
        optimizer.average()
        final_ppl,  _ = eval_loop(dev_loader, criterion_eval, best_model)
        optimizer.restore()
        writer.add_scalar('PPL/test', final_ppl, 1)


    # Save the best model
    torch.save(best_model.state_dict(), models_folder+'/'+exp_name+'.pt')

    writer.close()
