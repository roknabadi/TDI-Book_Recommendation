#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
import scipy.stats as stats
import numpy as np
from sklearn.metrics import accuracy_score
import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
import os

def _get_conv1d_out_length(l_in, padding, dilation, kernel, stride):
    return int((l_in + 2*padding - dilation*(kernel-1) - 1)/stride)+1

def accuracy(x, y):
    assert y.ndim == 1 and y.size() == x.size()
    x_p = torch.sigmoid(x)
    x_p = x_p >= 0.5
    return (x_p == y).sum().item() / y.size(0)
    
def accuracy2(pred, targets, thr=0.5):
    pred_p = sigm_on_pred(pred)
    accuracy = accuracy_score(targets, pred_p >= thr)
    return accuracy

def sigm_on_pred(pred):
    pred_p = torch.sigmoid(torch.from_numpy(pred))
    return pred_p

class BookRecommender(nn.Module):
    def __init__(self, books_size, users_size, books_dim=48,  users_dim=48, n_channel=8, 
                 conv1kc=1, conv1ks=10, conv1st=1, conv1pd=10, pool1ks=10, pool1st=5 , pdrop1=0.2, #conv_block_1 parameters
                 pdropbooks=0.5, pdropusers = 0.5,            
                 fchidden1 = 64,fchidden2 = 32, pdropfc=0.5, final=1, #fully connected parameters                 
                 opt="Adam", loss="BCELoss", lr=1e-3, momentum=0.9, weight_decay=1e-3
                ):
        super(BookRecommender, self).__init__()
        
        
        self.opt = opt
        self.loss = loss
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.lr = lr

        self.booklayer = nn.Sequential(
            nn.Embedding(books_size, books_dim),
            nn.Dropout(p=pdropbooks)
        )
        self.userlayer = nn.Sequential(
            nn.Embedding(users_size, users_dim),
            nn.Dropout(p=pdropusers)
        )
        
            
#        mpool_block_1_out_len=8
        
        self.conv_block_1 = nn.Sequential(
            nn.Conv1d(n_channel, conv1kc, kernel_size=conv1ks, stride=conv1st, padding=conv1pd),
            nn.LeakyReLU(),
            nn.BatchNorm1d(conv1kc),
            nn.MaxPool1d(kernel_size=pool1ks, stride=pool1st),
            nn.Dropout(p=pdrop1),
        )
#        max_len=books_dim+users_dim
#        conv_block_1_out_len = _get_conv1d_out_length(max_len, conv1pd, 1, conv1ks, conv1st) #l_in, padding, dilation, kernel, stride
#        mpool_block_1_out_len = _get_conv1d_out_length(conv_block_1_out_len, 0, 1, pool1ks, pool1st)
        mpool_block_1_out_len = books_dim+users_dim
        self.fc = nn.Sequential(
            nn.Linear(mpool_block_1_out_len, fchidden1),
            nn.ReLU(),
            nn.Dropout(p=pdropfc),
            nn.Linear(fchidden1, fchidden2),
            nn.ReLU(),
            nn.Dropout(p=pdropfc),
            nn.Linear(fchidden2, final)
        )
        
 



        
        
    def forward(self, books, users):
        embedbooks = self.booklayer(books)
        embedusers = self.userlayer(users)
        similarity=torch.cat([embedbooks, embedusers], dim=-1)   
        res = self.fc(similarity)
        res = torch.flatten(res)
        return res
        
                
    def compile(self, device='cpu'):
        self.to(device)
        if self.opt=="Adam":
            self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        if self.opt=="SGD":
            self.optimizer = optim.SGD(self.parameters(), lr=self.lr, momentum = self.momentum)
        if self.opt=="Adagrad":
            self.optimizer = optim.Adagrad(self.parameters(), lr=self.lr, weight_decay = self.weight_decay)
        if self.loss=="BCELoss":
            self.loss_fn = nn.BCEWithLogitsLoss()
            
    def train_model(self, train_loader, val_loader=None, epochs=100, clip=10, device='cpu', modelfile='models/best_ret_checkpoint.pt', logfile = None, tuneray=False, verbose=True):
        train_losses = []
        train_accs = []
        
        valid_losses = []
        valid_accs = []
        
        best_model=10000000
        for epoch in range(epochs):
            training_loss = 0.0
            train_acc = 0.0
            
            for batch_idx, batch in enumerate(train_loader):
                self.train()
                self.optimizer.zero_grad()
                books,users, ratings = batch
                books = books.to(device)
                users = users.to(device)
                out = self(books, users)
                loss = self.loss_fn(out.to('cpu'),ratings.to(torch.float32).to('cpu'))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), clip)
                self.optimizer.step()
                training_loss = loss.data.item() * books.size(0) 
                train_acc = accuracy(out.to('cpu'), ratings.to(torch.float32).to('cpu'))

                
                train_losses.append(training_loss)
                train_accs.append(train_acc)
                    
                if verbose and batch_idx%10==0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {:.6f}'.format(
                            epoch, batch_idx * train_loader.batch_size, len(train_loader.dataset), 100. * batch_idx * float(train_loader.batch_size) / len(train_loader.dataset),
                            loss.item(),  train_acc))
                    
            if val_loader:
                target_list = []
                pred_list = []
                valid_loss = 0.0
                valid_acc = 0.0
                
                self.eval()
                for batch_idx, batch in enumerate(val_loader):
                    books,users, ratings = batch
                    books = books.to(device)
                    users = users.to(device)
                    out = self(books, users)
                    loss = self.loss_fn(out.to('cpu'),ratings.to(torch.float32).to('cpu'))
                    valid_loss += loss.data.item() * books.size(0)
                    pred_list.append(out.to('cpu').detach().numpy())
                    target_list.append(ratings.to('cpu').detach().numpy())
                targets = np.concatenate(target_list)
                preds = np.concatenate(pred_list)
                valid_loss /= len(val_loader.dataset)
                valid_acc = accuracy2(preds, targets)

                valid_losses.append(valid_loss)
                valid_accs.append(valid_acc)
                
                if tuneray:
                    with tune.checkpoint_dir(step=epoch) as checkpoint_dir:
                        path = os.path.join(checkpoint_dir, "checkpoint")
                        torch.save(
                            (self.state_dict(), self.optimizer.state_dict()), path)

                    tune.report(loss=valid_loss, acc= valid_acc)

                if verbose:
                    print('Validation: Loss: {:.6f}\tAccuracy: {:.6f}'.format(valid_loss,  valid_acc))
                if logfile:
                    logfile.write('Validation: Loss: {:.6f}\tAccuracy: {:.6f}\n'.format(valid_loss,  valid_acc))

                if (valid_loss<best_model):
                    best_model = valid_loss
                    if modelfile:
                        print('Best model updated.')
                        self.save_model(modelfile)
                
                
        return {'train_loss':train_losses, 'train_accs':train_accs, 'valid_loss':valid_losses,  'valid_accs':valid_accs}
    
    def save_model(self, filename):
        torch.save(self.state_dict(), filename)
    
    def load_model(self, filename):
        self.load_state_dict(torch.load(filename))


