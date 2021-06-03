import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from buffer import Buffer
from utils import binary_accuracy


def train_cl(train_set,test_set, model, loss, optimizer, epochs,config,device):
    buffer = Buffer(config['buffer_size'], device)
    for index,data_set in enumerate(train_set): # train set is a list of N domain
        model.train()
        if index == 0: # PRIMO DOMINIO LO ALLENO "NORMALMENTE"
            train_loader = DataLoader(data_set, batch_size=config["batch_size"], shuffle=False)

            for i in tqdm(range(epochs)):
                loss_list = []
                acc_list = []
                for j, (x, y) in enumerate(train_loader):
                    optimizer.zero_grad()

                    x = x.cuda()
                    y_pred = model(x)

                    y = y.cuda()
                    s_loss = loss(y_pred.squeeze(1), y)
                    acc = binary_accuracy(y_pred.squeeze(1), y)
                    loss_list.append(s_loss.item())
                    acc_list.append(acc.item())

                    s_loss.backward()
                    optimizer.step()

            #DOPO AVERLO ALLENTO LO TESTO NORMALMENTE

        else: #GLI ALTRI DOMINI LI FACCIO CON IL BUFFER
            train_loader = DataLoader(data_set, batch_size=config["batch_size"] / 2, shuffle=False)
            for epoch in tqdm(range(epochs)):
                loss_list = []
                acc_list = []
                for i,(x,y) in enumerate(train_loader): #data_set già nel formato loader
                    optimizer.zero_grad()
                    inputs = x.cuda()
                    labels = y.cuda()

                    if not buffer.is_empty():
                        buf_input,buf_label = buffer.get_data(config['batch_size']/2) #Strategy 50/50
                        inputs = torch.cat((inputs,buf_input))
                        labels = torch.cat((labels,buf_label))

                    y_pred = model(inputs)
                    s_loss = loss(y_pred.squeeze(1),labels)
                    acc = binary_accuracy(y_pred.squeeze(1),labels)