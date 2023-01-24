#!/usr/bin/env python
# coding: utf-8

# In[1]:


from torch.optim import Adam
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from numpy import vstack
from numpy import sqrt
import pandas as pd
from sklearn.metrics import mean_squared_error
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch import Tensor
from torch.nn import Linear
from torch.nn import Sigmoid
from torch.nn import Module
from torch.optim import SGD
from torch.nn import MSELoss
from torch.nn import HuberLoss
from torch.nn.init import xavier_uniform_
from math import sqrt
import matplotlib.pyplot as plt


# In[2]:


Data = pd.read_csv('houses.csv')
Data.info()


# In[3]:


Data.isnull().sum()


# In[4]:


correlation_matrix = Data.corr()
correlation_matrix


# In[5]:


import seaborn as sns
hist_plot = sns.distplot(Data['price'])


# In[6]:


plt.scatter(x = Data['price'], y = Data['sqft_living'])


# In[7]:


year = []
month = []
for item in Data.date:
    year.append(int(item[:4]))
    month.append(int(item[4:6]))
Data['Year'] = year
Data['Month'] = month
Data_new = Data.drop('date', axis=1)
Data_new = Data_new.drop('id', axis=1)
y = Data_new['price']
x = Data_new.drop('price', axis=1)
dataa = x
x_scaler = MinMaxScaler()
x = x_scaler.fit_transform(x)
y = y.to_numpy()
y = y.reshape(1, -1)
y = np.log(y)


# In[8]:


x = pd.DataFrame(x)


# In[9]:


y = y.reshape(-1, 1)
x['target'] = y


# In[10]:


Data_new = x
Data_new.shape


# In[11]:


x['target']


# In[12]:


class HousesDataset(Dataset):
    # load the dataset
    def __init__(self, df):
        # store the inputs and outputs
        self.X = df.values[:, :-1].astype('float32')
        self.y = df.values[:, -1].astype('float32')
        # ensure target has the right shape
        self.y = self.y.reshape((len(self.y), 1))
 
    # number of rows in the dataset
    def __len__(self):
        return len(self.X)
 
    # get a row at an index
    def __getitem__(self, idx):
        return [self.X[idx], self.y[idx]]
 
    # get indexes for train and test rows
    def get_splits(self, n_test=0.2):
        # determine sizes
        test_size = round(n_test * len(self.X))
        train_size = len(self.X) - test_size
        # calculate the split
        return random_split(self, [train_size, test_size])


# In[13]:


# model definition
class MLP(Module):
    # define model elements
    def __init__(self, n_inputs):
        super(MLP, self).__init__()
        # input layer
        self.hidden1 = Linear(n_inputs, 32)
        xavier_uniform_(self.hidden1.weight)
        self.act1 = Sigmoid()
         # first hidden layer
        self.hidden2 = Linear(32, 32)
        xavier_uniform_(self.hidden2.weight)
        self.act2 = Sigmoid()
        # second hidden layer
        self.hidden3 = Linear(32, 8)
        xavier_uniform_(self.hidden3.weight)
        self.act3 = Sigmoid()
        #output layer
        self.hidden4 = Linear(8, 1)
        xavier_uniform_(self.hidden4.weight)
 
    # forward propagate input
    def forward(self, X):
        # input layer
        X = self.hidden1(X)
        X = self.act1(X)
        # first hidden layer
        X = self.hidden2(X)
        X = self.act2(X)
        #second hidden layer
        X = self.hidden3(X)
        X = self.act3(X)
        # third hidden layer and output
        X = self.hidden4(X)
        return X


# In[14]:


# prepare the dataset
def prepare_data(data):
    dataset = HousesDataset(data)
    # calculate split
    train, test = dataset.get_splits()
    # prepare data loaders
    train_dl = DataLoader(train, batch_size=1000, shuffle=True)
    test_dl = DataLoader(test, batch_size=1000, shuffle=False)
    
    return train_dl, test_dl


# In[15]:


# train the model
def train_model(train_dl, model, optim, criter, n_epochs):
    # define the optimization
    criterion = criter
    optimizer = optim
    # enumerate epochs
    train_losses = np.zeros(n_epochs)
    
    for epoch in range(n_epochs):
        # enumerate mini batches
        train_loss = []
        for i, (inputs, targets) in enumerate(train_dl):
            
            # clear the gradients
            optimizer.zero_grad()
            # compute the model output
            yhat = model(inputs)
            # calculate loss
            loss = criterion(yhat, targets)
            # credit assignment
            loss.backward()
            # update model weights
            optimizer.step()
            train_loss.append(loss.item())
         
        train_loss = np.mean(train_loss)
        train_losses[i] = train_loss
    return train_losses


# In[16]:


def Plot_MLP(train_dl, test_dl, optim, criter, n_epochs):
    #plot Loss and validationLoss plot
    criterion = criter
    optimizer = optim
    train_losses = np.zeros(n_epochs)
    test_losses = np.zeros(n_epochs)
    for it in range(n_epochs):
        train_loss = []
        for i, (inputs, targets) in enumerate(train_dl):

            optimizer.zero_grad( )
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())
        train_loss = np.mean(train_loss)
        test_loss = []
        for i, (inputs, targets) in enumerate(test_dl):
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss.append(loss.item())
        test_loss = np.mean(test_loss)
        train_losses[it] = train_loss
        test_losses[it] = test_loss

        print(f'epoch {it+1}/{n_epochs}, train loss: {train_loss:.4f}, test loss: {test_loss:.4f}')
    plt.semilogy(train_losses, label='train')
    plt.semilogy(test_losses, label='test')
    plt.legend()
    plt.show()
    #plt.plot(train_losses, label='train')
    #plt.plot(test_losses, label='test')
    #plt.legend()
    #plt.show()


# In[17]:


# evaluate the model
def evaluate_model(test_dl, model, criter):
    predictions = []
    actuals = []
    
    criterion = criter
    for i, (inputs, targets) in enumerate(test_dl):

        # evaluate the model on the test set
        yhat = model(inputs)
        output = yhat
        # retrieve numpy array
        yhat = yhat.detach().numpy()
        actual = targets.numpy()
        actual = actual.reshape((len(actual), 1))
        # store
        predictions.append(yhat)
        actuals.append(actual)


    predictions, actuals = vstack(predictions), vstack(actuals)
    # calculate mse
    mse = mean_squared_error(actuals, predictions)
    return mse


# In[18]:


# make a class prediction for one row of data
def predict(row, model):
    # convert row to data
    row = Tensor([row])
    # make prediction
    yhat = model(row)
    # retrieve numpy array
    yhat = yhat.detach().numpy()
    return yhat
for i in range(5):
    data = Data_new.loc[random.randint(0, 21613)]
    y = data['target']
    x = data.drop('target')
    yhat = predict(x, model)
    print('Predicted: %.3f' % yhat, y)


# In[24]:


# prepare the data
train_dl, test_dl = prepare_data(Data_new)
print(len(train_dl.dataset), len(test_dl.dataset))
# define the network
model = MLP(20)
#defin epochs
n_epochs = 2000
#defin optimizer and lossfunction
criterion1 = MSELoss()
optimizer1 = Adam(model.parameters(), lr=0.08)
criterion2 = MSELoss()
optimizer2 = SGD(model.parameters(), lr=0.001)
# train the model
train_model(train_dl, model, optimizer2, criterion2, n_epochs)
# evaluate the model
mse = evaluate_model(test_dl, model, criterion2)
print('MSE: %.3f, RMSE: %.3f' % (mse, sqrt(mse)))
import random
for i in range(5):
    data = Data_new.loc[random.randint(0, 21613)]
    y = data['target']
    x = data.drop('target')
    yhat = predict(x, model)
    print('distance of y and yhat: %.3f' %(y-yhat))
    print('Predicted: %.3f' % yhat, y)
Plot_MLP(train_dl, test_dl, optimizer2, criterion2, n_epochs)


# In[ ]:




