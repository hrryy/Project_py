import torch
import torch.nn.init

from test_util import *  # *: all     import *: all function
from train_util import *

def test(trained_model):
    device, model = test_model(run_mode,trained_model)
    print(testing(1,model,device))



def go_train():
    epochs, batch_size, learning_rate = configuration()
    select = input("[Select Model]\n\n [1] CNNet \n [2] CNNet Convolution Transform \n\n Select : ")
    device, model = train_model(select=select)

    trained_model = training(epochs, batch_size, learning_rate, model, device)
    if select == '1':
        torch.save(trained_model,f"{os.path.dirname}/trained_model/CNNet_{epochs}_{learning_rate}.pth")
    elif select == '2':
        torch.save(trained_model,f"{os.path.dirname}/trained_model/CNNet_convT_{epochs}_{learning_rate}.pth")
                
    print("<===================== Train and Save the Model Successfully =======================>\n")
    if go_test():
        test(trained_model)

#RUN MODE
print(""" 
    [1] Training and Test                      
    [2] Just Training
    [3] Just Test (pre-trained model required)
    [4] Exit
""")

run_mode = input("Select Run Mode : ")
if run_mode == '1' or run_mode == '2':
    go_train()
elif run_mode == '3':
    test()
elif run_mode == '4':
    print("Bye :-)")
    sys.exit()
else:
    print("input error")
    sys.exit()