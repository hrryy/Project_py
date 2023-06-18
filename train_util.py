import torch
import torch.nn.init
from torch.utils.data import DataLoader

import os # file path
import sys #sys.exit()

from dataset.data_preset import AlzheimerDataset
import model.CNNet as structure
import matplotlib.pyplot as plt

################
# Configuration #
################

def configuration():
    print("====== Setting to train your model ======")
    epochs = int(input("Epoch : "))
    batch_size = int(input("Batch_size : "))
    learning_rate = float(input("Learning Rate : "))

    return epochs, batch_size, learning_rate

########################
# Ready to Train Model #
########################

def train_model(select):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    if select == '1':
        model = structure.CNNet().to(device)
        print(model)
    elif select == '2':
        model = structure.CNNet_convtrans().to(device)
        print(model)

    return device,model

########################
# Design Training Flow #
########################

def training(epochs, batch_size, learning_rate, model, device):

    # config model
    criterion = torch.nn.CrossEntropyLoss().to(device) # with softmax func in cost function
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    losses = []

    model.train()                   #when test, model.eval()


    #get dataset
    current_path = os.path.dirname(__file__)

    input_path = sorted(os.listdir(f"{current_path}/dataset/train/"))
    input_path = [f'{current_path}/dataset/train/{paths}' for paths in input_path]

    dataset = AlzheimerDataset(input_path)
    data_loader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            drop_last=False)

    total_batch = len(data_loader)
    print('Total batch size : {}'.format(total_batch))

    for epoch in range(epochs):   # epoch is cycles
        avg_cost = 0   #average cost

        for X,Y in data_loader:   #X: img, Y: label
            
            X = X.to(device)
            Y = Y.to(device)

            optimizer.zero_grad()

            output = model(X)
            cost = criterion(output, Y)

            cost.backward()
            optimizer.step()

            
            print(cost.item())

            losses.append(cost.item())

        print("---Epoch : {:4d} cost : {:8} ---".format(epoch + 1, cost.item()))

    plt.plot(losses)
    plt.show()

    return model

def go_test():
    test_select = input("Do you want to test model right now? (y/n) : ")
    if test_select == 'y':
        return True
    else:
        sys.exit()



