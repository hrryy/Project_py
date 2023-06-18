import torch
import torch.nn.init
from torch.utils.data import DataLoader

import os # file path
import sys #sys.exit()

from dataset.data_preset import AlzheimerDataset
from model.CNNet import CNNet

#######################
# Ready To Test Model #
#######################

def test_model(run_mode,trained_model):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"device : {device}")

    if(run_mode == '3'):
        trained_path = input("If you already ahve pretrained model, input path: ")
    else:
        trained_path = 'nothing'
    
    if os.path.isfile(trained_path):
        model = torch.load(trained_path).to(device)
    else:
        model = trained_model.to(device)
    
    print(model)
    return device, model

########################
# Design Test Flow #
########################

def testing(batch_size, model, device):

    
    model.eval()                   #when test, model.eval()


    #get dataset
    current_path = os.path.dirname(__file__)

    input_path = sorted(os.listdir(f"{current_path}/dataset/test/"))
    input_path = [f'{current_path}/dataset/test/{paths}' for paths in input_path]

    dataset = AlzheimerDataset(input_path)
    data_loader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            drop_last=False)

    total_batch = len(data_loader)
    print('Total batch size : {}'.format(total_batch))

    with torch.no_grad():

        i = 0
        correct_prediction = 0

        for X,Y in data_loader:   #X: img, Y: label
            
            X_test = X.to(device)
            Y_test = Y.to(device)

            prediction = model(X_test)
            correct_prediction += torch.argmax(prediction, 1) == Y_test
            i += 1

        accuracy = float(correct_prediction)/float(i)*100
        print("---accuracy : {:.4} % ---".format(accuracy))

    return 'Done'
