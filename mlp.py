import numpy as np
import pandas as pd
import os
from sklearn.metrics import accuracy_score
import sklearn.metrics as metrics
import pickle
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import scipy
import random
from IPython.utils.path import ensure_dir_exists
import json
from numpy import genfromtxt
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import shutil


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

root_path = "/u/erdos/students/mwiecksosa/" # change this to your directory

cough_path = root_path + "Cough_Analytics/"
ensure_dir_exists(cough_path)

data_path = cough_path + "Data/"
ensure_dir_exists(data_path)

save_model_path = root_path + "Cough_Analytics/Models/"
ensure_dir_exists(save_model_path)


with open(data_path + "all_patient_keys.txt",'rb') as f:
    data = f.readlines()

all_patient_keys = data[0]
#all_patient_keys = list(cough_dict.keys()) + list(snore_dict.keys())
print(all_patient_keys)
print("got all keys?",len(all_patient_keys))


class SimpleDataset(Dataset):
    def __init__(self, X, y):
        self.X, self.y = X, y
    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]).float(), torch.tensor([self.y[idx]]).float()
    def __len__(self):
        return len(self.X)

def get_train_val_test(val_patientID, testID_folder, batch_size=64):

    test_patientID = testID_folder[:2] #first two elems is patient ID

    Xtr =  pd.read_csv(data_path+testID_folder+"/"+"train_"+test_patientID,delimiter=',')
    Xte =  pd.read_csv(data_path+testID_folder+"/"+"test_"+test_patientID,delimiter=',')

    #dataID in form 'expl_away_noSNR_44_20170707_141339#3#4'

    # fileName col data in form: 44_20170707_141339#3#4
    Xva = Xtr[Xtr['fileName'].str.startswith(val_patientID)]
    Xtr = Xtr[~Xtr['fileName'].str.startswith(val_patientID)]

    # y target values
    ytr = Xtr.iloc[:,43]
    ytr = ytr.values

    yva = Xva.iloc[:,43]
    yva = yva.values

    yte = Xte.iloc[:,43]
    yte = yte.values

    #get rid of unnamed column and ID column
    Xtr = Xtr.iloc[:,2:43] #get rid of first 2 columns and just get values
    Xtr = Xtr.values

    Xva = Xva.iloc[:,2:43] #get rid of first 2 columns and just get values
    Xva = Xva.values

    Xte = Xte.iloc[:,2:43] #get rid of first 2 columns and just get values
    Xte = Xte.values


    #### ok

    tr = SimpleDataset(Xtr, ytr)
    va = SimpleDataset(Xva, yva)
    te = SimpleDataset(Xte, yte)

    tr_loader = DataLoader(tr, batch_size=batch_size, shuffle=True)
    va_loader = DataLoader(va, batch_size=batch_size)
    te_loader = DataLoader(te, batch_size=batch_size)

    print('Feature shape, Label shape, Class balance:')
    print("TR:",'\t', tr_loader.dataset.X.shape, tr_loader.dataset.y.shape, tr_loader.dataset.y.mean())
    if va_loader.dataset.y.size != 0:
        print("VA:",'\t', va_loader.dataset.X.shape, va_loader.dataset.y.shape, va_loader.dataset.y.mean())
    else:
        print("VA:",'\t', va_loader.dataset.X.shape, va_loader.dataset.y.shape, "empty!!!")
    print("TE:",'\t', te_loader.dataset.X.shape, te_loader.dataset.y.shape, te_loader.dataset.y.mean())

    return tr_loader, va_loader, te_loader






def get_train_test(testID_folder, batch_size=64):

    test_patientID = testID_folder[:2] #first two elems is patient ID

    Xtr =  pd.read_csv(data_path+testID_folder+"/"+"train_"+test_patientID,delimiter=',')
    Xte =  pd.read_csv(data_path+testID_folder+"/"+"test_"+test_patientID,delimiter=',')

    # y target values
    ytr = Xtr.iloc[:,43]
    ytr = ytr.values

    yte = Xte.iloc[:,43]
    yte = yte.values

    #get rid of unnamed column and ID column
    Xtr = Xtr.iloc[:,2:43] #get rid of first 2 columns and just get values
    Xtr = Xtr.values

    Xte = Xte.iloc[:,2:43] #get rid of first 2 columns and just get values
    Xte = Xte.values

    tr = SimpleDataset(Xtr, ytr)
    te = SimpleDataset(Xte, yte)

    tr_loader = DataLoader(tr, batch_size=batch_size, shuffle=True)
    te_loader = DataLoader(te, batch_size=batch_size)

    print('Feature shape, Label shape, Class balance:')
    print("TR:",'\t', tr_loader.dataset.X.shape, tr_loader.dataset.y.shape, tr_loader.dataset.y.mean())
    print("TE:",'\t', te_loader.dataset.X.shape, te_loader.dataset.y.shape, te_loader.dataset.y.mean())

    return tr_loader, te_loader

class MLP(nn.Module):
    def __init__(self,
        depth,
        input_dim,
        hidden_L1_dim,
        hidden_L2_dim,
        activation,
        dropout_prob
    ):


        super(MLP,self).__init__()

        self.input_dim = input_dim
        self.depth = depth


        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'elu':
            self.activation = F.elu

        if depth is 1:
            self.lin1 = nn.Linear(input_dim, 1, bias=True)
        elif depth is 2:
            self.lin1 = nn.Linear(input_dim, hidden_L1_dim, bias=True)
            self.lin2 = nn.Linear(hidden_L1_dim, 1, bias=True)
        elif depth is 3:
            self.lin1 = nn.Linear(input_dim, hidden_L1_dim, bias=True)
            self.lin2 = nn.Linear(hidden_L1_dim, hidden_L2_dim, bias=True)
            self.lin3 = nn.Linear(hidden_L2_dim, 1, bias=True)

    def forward(self, xb):

        x = xb.view(-1,self.input_dim)

        if self.depth == 1:
            x = self.activation(self.lin1(x))
        elif self.depth == 2:
            x = self.activation(self.lin1(x))
            x = self.activation(self.lin2(x))
        elif self.depth == 3:
            x = self.activation(self.lin1(x))
            x = self.activation(self.lin2(x))
            x = self.activation(self.lin3(x))

        x = torch.sigmoid(x)

        return x

def choose_hyperparameters():

  ranges = {
    "depth": [1,2,3],
    "hidden_L1_dim": [25,30,35,40],
    "hidden_L2_dim": [10,15,20,25],
    "dropout_prob": [0,0.1,0.2],
    "activation": ["relu","elu"],
    "cost_function": ["BCELoss"],
    "optimizer": ["Adam", "SGD"],
    "learning_rate_adam": [0, 0.0002],
    "learning_rate_sgd": [0, 0.02],
  }

  depth = random.choice(ranges['depth'])
  dropout_prob = random.choice(ranges['dropout_prob'])
  hidden_L1_dim = random.choice(ranges['hidden_L1_dim'])
  hidden_L2_dim = random.choice(ranges['hidden_L2_dim'])
  activation = random.choice(ranges['activation'])
  cost_function = random.choice(ranges['cost_function'])
  optimizer = random.choice(ranges['optimizer'])
  if optimizer == 'SGD':
    learning_rate = random.uniform(*ranges['learning_rate_sgd'])
  elif optimizer == 'Adam':
    learning_rate = random.uniform(*ranges['learning_rate_adam'])

  hyperparameters = {
    "depth": depth,
    "dropout_prob": dropout_prob,
    "hidden_L1_dim": hidden_L1_dim,
    "hidden_L2_dim": hidden_L2_dim,
    "activation": activation,
    "cost_function": cost_function,
    "optimizer": optimizer,
    "learning_rate": learning_rate,
  }

  return hyperparameters




def get_model_from_file(model_param_txt):

    with open(model_param_txt) as json_file:
      best_model_params = json.load(json_file)

    best_model =  MLP(
        depth = best_model_params['depth'],
        input_dim = int(41), #dropped ID column
        hidden_L1_dim = int(best_model_params['hidden_L1_dim']),
        hidden_L2_dim = int(best_model_params['hidden_L2_dim']),
        activation = best_model_params['activation'],
        dropout_prob = int(best_model_params['dropout_prob'])
    )

    best_model = best_model.to(device)

    cost_function = best_model_params['cost_function']
    if cost_function == "BCELoss":
      criterion = torch.nn.BCELoss()

    optimizer = best_model_params['optimizer']
    if optimizer == "SGD":
      optimizer = torch.optim.SGD(best_model.parameters(), lr = best_model_params['learning_rate'])

    elif optimizer == "Adam":
      optimizer = torch.optim.Adam(best_model.parameters(), lr = best_model_params['learning_rate'])


    return best_model, criterion, optimizer

for patientID_folder in os.listdir(data_path): #loop through test splits

    if not patientID_folder[0].isdigit():
        continue

    print(patientID_folder)

    testID_folder = patientID_folder

    test_patientID = testID_folder[:2] #first two elems is patient ID


    train_patientID_list = []

    for patientID in all_patient_keys:

        if patientID is not test_patientID:

            train_patientID_list.append(patientID)

    num_networks = 50 # create 50 models to try in hyperparameter selection


    save_test_split_dir = save_model_path + "test_"+test_patientID+"/"
    ensure_dir_exists(save_test_split_dir)

    model_hyperparam_95_conf_interval_AUROC_dict = dict()

    max_AUROC_of_all_hyperParamCombos_list = []

    for count in range(num_networks):

        max_AUROC_of_this_hyperparam_combo = []

        #identifier = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        identifier = test_patientID + "_test"
        model_params = choose_hyperparameters()
        model_params['batch_size'] = 64
        #tr_loader, va_loader, te_loader = get_train_val_test(batch_size=64)

        save_hyper_param_dir = save_test_split_dir + 'model_' + identifier + '_' + str(count+1) + "_hyperparam"+ '_created/'
        model_params['save_dir'] = save_hyper_param_dir

        ensure_dir_exists(model_params['save_dir'])
        with open(model_params['save_dir'] + 'model_params'+'.txt','w') as file:
          file.write(json.dumps(model_params))

        n_epochs = 30 #for early stopping


        #depth variable, 1, 2, 3, number of hidden layers in NN

        model =  MLP(
            depth = model_params['depth'],
            input_dim = int(41), #dropped ID column
            hidden_L1_dim = int(model_params['hidden_L1_dim']),
            hidden_L2_dim = int(model_params['hidden_L2_dim']),
            activation = model_params['activation'],
            dropout_prob = int(model_params['dropout_prob'])
        )

        model = model.to(device)
        cost_function = model_params['cost_function']
        if cost_function == "BCELoss":
          criterion = torch.nn.BCELoss()
        optimizer = model_params['optimizer']
        if optimizer == "SGD":
          optimizer = torch.optim.SGD(model.parameters(), lr = model_params['learning_rate'])
        elif optimizer == "Adam":
          optimizer = torch.optim.Adam(model.parameters(), lr = model_params['learning_rate'])


        train_losses_list = []
        val_losses_list = []
        train_scores_list = []
        val_scores_list = []


        #get validation set, hyperparam optimize on val, test, get AUROC range, get median AUROC

        for train_patientID in train_patientID_list:


            outputs = [] #only compare max perf of each val split

            val_patientID = train_patientID

            tr_loader, va_loader, te_loader = get_train_val_test(val_patientID = val_patientID, testID_folder=testID_folder,batch_size=64)


            if va_loader.dataset.X.shape[0] is 0 or va_loader.dataset.y.shape[0] is 0:
                continue



            #print("got the data loaders")

            save_val_split_dir = save_hyper_param_dir + 'model_' + identifier + '_' + val_patientID + "_" + "val" + "_" + str(count+1) + '_created/'

            ensure_dir_exists(save_val_split_dir)

            print("%s as test, %i hyperparam num, %s as val"%(test_patientID,count,val_patientID))

            ########################### Epoch 0
            print('Epoch', 0)

            #### evaluate model
            model.eval()
            with torch.no_grad():
                # Evaluate on train
                y_true, y_score = [], []
                running_loss = []
                for X, y in tr_loader:
                    X, y = X.to(device), y.to(device)
                    output = model(X)
                    y_true.append(y.cpu().numpy())
                    y_score.append(output.cpu().numpy())
                    running_loss.append(criterion(output, y).item())

                y_true, y_score = np.concatenate(y_true), np.concatenate(y_score)
                train_loss = np.mean(running_loss)
                train_score = metrics.roc_auc_score(y_true, y_score)
                print('tr loss', train_loss, 'tr AUROC', train_score)

                # Evaluate on validation
                y_true, y_score = [], []
                running_loss = []
                for X, y in va_loader:
                    X, y = X.to(device), y.to(device)
                    with torch.no_grad():
                        output = model(X)
                        y_true.append(y.cpu().numpy())
                        y_score.append(output.cpu().numpy())
                        running_loss.append(criterion(output, y).item())


                y_true, y_score = np.concatenate(y_true), np.concatenate(y_score)
                val_loss = np.mean(running_loss)
                val_score = metrics.roc_auc_score(y_true, y_score)
                print('va loss', val_loss, 'va AUROC', val_score)
                #### end evaluate model

            outputs.append((train_loss, val_loss, train_score, val_score))

            ########################### Epochs 1-29
            for epoch in range(0, n_epochs):
                print("epoch",epoch)

                #### train model

                model.train()

                for X, y in tr_loader:
                    #print("in train loader loop")
                    X, y = X.to(device), y.to(device)
                    # clear parameter gradients
                    optimizer.zero_grad()
                    # forward + backward + optimize
                    output = model(X)
                    loss = criterion(output, y)
                    loss.backward()
                    optimizer.step()
                #### end train model

                #### evaluate model
                model.eval()
                with torch.no_grad():
                    # Evaluate on train
                    y_true, y_score = [], []
                    running_loss = []
                    for X, y in tr_loader:
                        X, y = X.to(device), y.to(device)
                        output = model(X)
                        y_true.append(y.cpu().numpy())
                        y_score.append(output.cpu().numpy())
                        running_loss.append(criterion(output, y).item())

                    y_true, y_score = np.concatenate(y_true), np.concatenate(y_score)
                    train_loss = np.mean(running_loss)
                    train_score = metrics.roc_auc_score(y_true, y_score)
                    print('tr loss', train_loss, 'tr AUROC', train_score)

                    # Evaluate on validation
                    y_true, y_score = [], []
                    running_loss = []
                    for X, y in va_loader:
                        X, y = X.to(device), y.to(device)
                        with torch.no_grad():
                            output = model(X)
                            y_true.append(y.cpu().numpy())
                            y_score.append(output.cpu().numpy())
                            running_loss.append(criterion(output, y).item())


                    y_true, y_score = np.concatenate(y_true), np.concatenate(y_score)
                    val_loss = np.mean(running_loss)
                    val_score = metrics.roc_auc_score(y_true, y_score)
                    print('va loss', val_loss, 'va AUROC', val_score)
                #### end evaluate model

                #### save outputs
                outputs.append((train_loss, val_loss, train_score, val_score))
                #### end save outputs

                #### save model parameters
                ensure_dir_exists(save_val_split_dir+'checkpoint/')
                state = {
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                }
                filename = os.path.join(save_val_split_dir+'checkpoint/', 'epoch={}.checkpoint.pth.tar'.format(epoch+1))
                torch.save(state, filename)
                #### end save model parameters


            print("outputs:",outputs)
            train_losses, val_losses, train_scores, val_scores = zip(*outputs)
            print("val scores:", val_scores)
            print("val losses:", val_losses)
            train_losses_list.append(train_losses)
            val_losses_list.append(val_losses)
            train_scores_list.append(train_scores)
            val_scores_list.append(val_scores)

        #get mean AUROC for each epoch, across all val splits, for the particular hyper param combo

        print("val_scores_list",val_scores_list)
        avg_val_scores_list = np.zeros(len(val_scores_list[0]))
        for i_list in val_scores_list:
          avg_val_scores_list += np.asarray(i_list)
        avg_val_scores_list = avg_val_scores_list / len(val_scores_list)
        #print("val_scores_list",avg_val_scores_list)

        #print("avg_val_losses_list",avg_val_losses_list)
        avg_val_losses_list = np.zeros(len(val_losses_list[0]))
        for i_list in val_losses_list:
          avg_val_losses_list += np.asarray(i_list)
        avg_val_losses_list = avg_val_losses_list / len(val_losses_list)
        #print("avg_val_losses_list",avg_val_losses_list)


        avg_train_losses_list = np.zeros(len(train_losses_list[0]))
        for i_list in train_losses_list:
          avg_train_losses_list += np.asarray(i_list)
        avg_train_losses_list = avg_train_losses_list / len(train_losses_list)
        #print(avg_train_losses_list)

        avg_train_scores_list = np.zeros(len(train_scores_list[0]))
        for i_list in train_scores_list:
          avg_train_scores_list += np.asarray(i_list)
        avg_train_scores_list = avg_train_scores_list / len(train_scores_list)



        max_AUROC_dict = dict()
        max_AUROC_dict[save_hyper_param_dir] = max(avg_val_scores_list) #max AUROC for particular hyperParamCombo, avg on validation sets, of all 30 epochs
        max_AUROC_of_all_hyperParamCombos_list.append(max_AUROC_dict)

        with open(save_hyper_param_dir + 'max_AUROC_of_this_hyperparam_combo'+'.txt','w') as file:
          file.write(json.dumps(max_AUROC_dict))


        fig, ax = plt.subplots(figsize=(5,5))
        plt.plot(range(n_epochs + 1), avg_train_scores_list, '--o', label='Train')
        plt.plot(range(n_epochs + 1), avg_val_scores_list, '--o', label='Validation')
        plt.xlabel('epoch')
        plt.ylabel('AUROC')
        plt.legend()
        plt.savefig(model_params['save_dir']+'auroc.png', dpi=300)

        fig, ax = plt.subplots(figsize=(5,5))
        plt.plot(range(n_epochs + 1), avg_train_losses_list, '--o', label='Train')
        plt.plot(range(n_epochs + 1), avg_val_losses_list, '--o', label='Validation')
        plt.xlabel('epoch')
        plt.ylabel('Loss (binary cross entropy)')
        plt.legend()
        plt.savefig(model_params['save_dir']+'loss.png', dpi=300)

    #loops through the mean val AUROC to find the best hyperparam combo of this test split
    max_AUROC_soFar = 0
    for adict in max_AUROC_of_all_hyperParamCombos_list:
        for key in adict.keys():
            if max_AUROC_soFar < adict[key]:
                max_AUROC_soFar = adict[key]
                max_hyperParam_path = key #gives save_hyper_param_dir


    #test on the test set with the full training set
    #get the test AUROC for all test splits

    best_hyperParam_dir = save_test_split_dir + "best_model_hyperParamCombo/"
    ensure_dir_exists(best_hyperParam_dir)

    with open(best_hyperParam_dir + 'max_AUROC'+'.txt','w') as file:
        file.write(str(max_AUROC_soFar))
    with open(best_hyperParam_dir + 'best_hyperParamCombo_path'+'.txt','w') as file:
        file.write(str(max_hyperParam_path))

    shutil.copy(max_hyperParam_path+"loss.png",best_hyperParam_dir+"best_loss.png")
    shutil.copy(max_hyperParam_path+"auroc.png",best_hyperParam_dir+"best_auroc.png")


    model_params_path = max_hyperParam_path + "model_params.txt"

    with open(model_params_path) as json_file:
        best_model_params = json.load(json_file)

    with open(best_hyperParam_dir + 'best_model_params'+'.txt','w') as file:
        file.write(json.dumps(best_model_params))

    best_model, criterion, optimizer = get_model_from_file(model_params_path)

    #get the train test split, on the entire train split, no validation split b/c used cross validation!
    tr_loader, te_loader = get_train_test(testID_folder=testID_folder,batch_size=64)

    for epoch in range(0, n_epochs): # 30 epochs

        print("epoch",epoch)

        #### train model

        best_model.train()

        for X, y in tr_loader:
            #print("in train loader loop")
            X, y = X.to(device), y.to(device)
            # clear parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            output = best_model(X)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
        #### end train model


        #### save model parameters
        ensure_dir_exists(best_hyperParam_dir+'checkpoint/')
        state = {
            'epoch': epoch,
            'state_dict': best_model.state_dict(),
        }
        filename = os.path.join(best_hyperParam_dir+'checkpoint/', 'epoch={}.checkpoint.pth.tar'.format(epoch+1))
        torch.save(state, filename)
        #### end save model parameters
