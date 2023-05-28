# data preset used to pull array from image and label no,mild,very mild, moderate
from torch.utils.data import Dataset
from torchvision import transforms
import glob
from PIL import Image

class AlzheimerDataset(Dataset):
    def __init__(self,path):
        self.path = path
        self.img_list = []
        self.label_list = []

        self.no_path = path[3]
        self.verymild_path = path[4]
        self.mild_path = path[1]
        self.moderate_path = path[2]

        self.no_img_list = glob.glob(self.no_path + '/*.jpg')
        self.verymild_img_list = glob.glob(self.verymild_path + '/*.jpg')
        self.mild_img_list = glob.glob(self.mild_path + '/*.jpg')
        self.moderate_img_list = glob.glob(self.moderate_path + '/*.jpg')

        self.transform = transforms.Compose([transforms.ToTensor(),])

        self.img_list = self.no_img_list + self.verymild_img_list + self.mild_img_list + self.moderate_img_list
        self.label_list = [0]*len(self.no_img_list)+ [1]*len(self.verymild_img_list) + [2]*len(self.mild_img_list) + [3]*len(self.moderate_img_list)
        # no : 0
        # very mild : 1
        # mild : 2
        # moderate : 3

    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self,idx):
        img_path = self.img_list[idx]
        label = self.label_list[idx]
        img = Image.open(img_path)  # image change to array 

        if self.transform is not None:
            img = self.transform(img)

        return img, label    

