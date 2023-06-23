import torch
import torchvision
import os
from torch.utils.data import Dataset
from PIL import Image
from tqdm import tqdm
import pathlib
import pandas as pd

class MarioHistoryDataset(Dataset):
    def __init__(self,img_dir,metadata_file,history_frames:int = 7,use_color=False,preload = False):
        super().__init__()
        self.history_frames = history_frames
        self.img_dir = img_dir
        self.use_color = use_color
        self.metadata = pd.read_csv(metadata_file)
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((256,256)),
            #torchvision.transforms.Grayscale(num_output_channels=1),
            torchvision.transforms.ToTensor(),
            
            #torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        self.total_length = len(self.metadata)# - self.history_frames + 1
        self.shape = (256,256)#Image.open(os.path.join(episode_dir,self.file_names[0])).convert("L").size
    def __len__(self):
        return self.total_length
    def __getitem__(self, idx):
        item = torch.zeros((self.history_frames,self.shape[1],self.shape[0]))
        for i in range(self.history_frames):
            current_img = self._get_image(idx+i)
            #print(f"shape:{current_img.shape},shape single:{current_img[0].shape}")
            if not self.use_color:
                item[i] = current_img
            else:
                item[3*i] =  current_img[0]
                item[3*i+1] =  current_img[1]
                item[3*i+2] =  current_img[2]
                
            
        return item, self._extract_action(idx)
    
    def _get_image(self,idx,offset=0):
        metadata_img = self.metadata.iloc[idx]
        current_img = Image.open(os.path.join(self.img_dir,f"{int(metadata_img['id'])-offset}.jpg"))
        if not self.use_color:
            current_img = current_img.convert("L")
        else:
            current_img = current_img.convert("RGB")
        if self.transform:
            current_img = self.transform(current_img)
        return current_img
    def _extract_action(self, idx):
        metadata_img = self.metadata.iloc[idx]
        action_tensor = torch.from_numpy(metadata_img[['up','left','right','B']].values)
        return action_tensor
        
