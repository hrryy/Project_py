import torch
import torch.nn.init

from test_util import *  # *: all     import *: all function
from train_util import *

def test():
    device, model = test_model(run_mode)
    print(testing(1,model,device))



def go_train():
    epochs, batch_size, learning_rate = configuration()
    device, model = train_model()

    torch.save(training(epochs, batch_size, learning_rate, model, device),
                f"{os.path.dirname}/trained_model/CNNet_{epochs}_{learning_rate}.pth")
    print("<===================== Train and Save the Model Successfully =======================>\n")
    if go_test():
        test()

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