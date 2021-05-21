import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import statistics
import numpy as np
import os.path as osp


class SimpleMLP(nn.Module):
    def __init__(self,input_size,hid_size):
        super(SimpleMLP,self).__init__()
        self.input_size = input_size
        self.hid_size = hid_size
        self.net = nn.Sequential(
            nn.Linear(input_size, hid_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hid_size, hid_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hid_size, hid_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hid_size, 1),
        )

    def forward(self,x):
        x = self.net(x)
        return x


def split_seq(sequence,n_steps):

    inout_seq = []
    for i in range(len(sequence)):
        end = i+n_steps
        if end> (len(sequence)-1):
            break
        seq_x,seq_y= sequence[i:end],sequence[end]
        inout_seq.append((seq_x,seq_y))
    return inout_seq


# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)


def read_csv():
    ROOT_PATH = ""
    csv_path = osp.join(ROOT_PATH, 'test-brent.csv')
    lines = [x.strip() for x in open(csv_path, 'r').readlines()][1:]
    value_list = lines[0].split(',')
    value_list.pop(0)
    value_list = [float(i) for i in value_list]
    return value_list


def train_model(data,model,loss,optimizer,epochs):
    print("Preparing data...")
    mse = []
    model.train()
    for i in range(epochs+1):
        loss_list = []
        printProgressBar(i + 1,epochs, prefix='Progress:', suffix='Complete', length=50)
        for seq, label in data:
            optimizer.zero_grad()

            y_pred = model(seq)

            s_loss = loss(y_pred[0], label)
            loss_list.append(s_loss.item())
            s_loss.backward()

            optimizer.step()

        if i % 50 == 0:
            print(f'epoch: {i:3} loss: {statistics.mean(loss_list)}')

        mse.append(statistics.mean(loss_list))

    plt.plot(mse)
    plt.show()

    torch.save(model.state_dict(),'model.pt')


def test_model(data,model,loss):
    model.load_state_dict(torch.load('model.pt'))

    model.eval()
    predicted = []
    test_loss = []
    for seq, label in data:
        with torch.no_grad():
            pred = model(seq)
            s_loss = loss(pred[0], label)
            test_loss.append(s_loss.item())
            predicted.append(pred[0].numpy())
            # print(f"Predicted: {pred}, loss: {s_loss.item()}")

    print(f"Test error: {statistics.mean(test_loss)}")

    return predicted


def main():
    raw_seq = read_csv()
    raw_seq = torch.Tensor(raw_seq).view(-1)
    data =split_seq(raw_seq,4)
    train_data = data[:1400]
    test_data = data[1400:]
    # for seq,label in data:
    #     print(seq,label)

    # Setup asnd train the model
    model = SimpleMLP(input_size=4,hid_size=100)
    loss = nn.MSELoss()
    optimizer = torch.optim.Adadelta(model.parameters(), lr=0.01)
    epochs = 200
    print(model)
    #train_model(train_data,model=model,loss=loss,optimizer=optimizer,epochs=epochs)



    # Test phase
    predicted = test_model(test_data,model=model,loss=loss)

    plt.plot(raw_seq, label='Ground truth')
    x = np.arange(1400, 1734, 1)
    plt.plot(x, predicted, label='Predicted')
    plt.autoscale(axis='x', tight=True)
    plt.show()

    plt.plot(x, raw_seq[-334:], label='Ground truth')
    plt.plot(x, predicted, label='Predicted')
    plt.autoscale(axis='x', tight=True)
    plt.show()


if __name__=="__main__":
    main()