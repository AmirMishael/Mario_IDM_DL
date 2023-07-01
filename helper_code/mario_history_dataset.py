import torch
import torchvision
import os
from torch.utils.data import Dataset
from PIL import Image
from tqdm import tqdm
import pathlib
import pandas as pd

from helper_code.mario_buttons_dataset import TEST_WORLDS, TRAIN_WORLDS, MarioButtonsDataset, MarioEpisode

class MarioHistoryDataset(Dataset):
    def __init__(self,img_dir,metadata_file,history_frames:int = 7,use_color=False,preload = False):
        super().__init__()
        self.history_frames = history_frames
        self.img_dir = img_dir
        self.use_color = use_color
        self.metadata = pd.read_csv(metadata_file,dtype={'id':int,'image_path':str,'up':float,'left':float,'right':float,'B':float}) #columns=['id,image_path','up','left','right','B']
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((256,256)),
            #torchvision.transforms.Grayscale(num_output_channels=1),
            torchvision.transforms.ToTensor(),
            
            #torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        self.total_length = len(self.metadata)-10# - self.history_frames + 1
        self.shape = (256,256)#Image.open(os.path.join(episode_dir,self.file_names[0])).convert("L").size
    def __len__(self):
        return self.total_length
    def __getitem__(self, idx):
        item = torch.zeros((self.history_frames,self.shape[1],self.shape[0]))
        for i in range(self.history_frames):
            current_img = self._get_image(idx,i)
            #print(f"shape:{current_img.shape},shape single:{current_img[0].shape}")
            if not self.use_color:
                item[i] = current_img
            else:
                item[3*i] =  current_img[0]
                item[3*i+1] =  current_img[1]
                item[3*i+2] =  current_img[2]
                
            
        return item, self._extract_action(idx),f"{1}-{1}"
    
    def _get_image(self,idx,offset=0):
        metadata_img = self.metadata.iloc[idx]
        img_path = os.path.join(self.img_dir,f"{int(metadata_img['id'])-offset}.jpg")
        #print(f"loading image :{img_path}")
        current_img = Image.open(img_path)
        if not self.use_color:
            current_img = current_img.convert("L")
        else:
            current_img = current_img.convert("RGB")
        if self.transform:
            current_img = self.transform(current_img)
        return current_img
    def _extract_action(self, idx):
        metadata_img = self.metadata.iloc[idx]
        #print(metadata_img[['up','left','right','B']].values.tolist())
        #print(metadata_img[['up','left','right','B']].values.tolist())
        #print(type(metadata_img[['up','left','right','B']].values))
        action_tensor = torch.round(torch.Tensor(metadata_img[['up','left','right','B']].values.tolist()))
        return action_tensor
        
class MarioHistoryEpisode(MarioEpisode):
    def __init__(self, episode_dir, group_frames: int = 1, use_color=False, transform=None, preload=False):
        super().__init__(episode_dir, group_frames, use_color, transform, preload)
    def __len__(self):
        return self.total_frames - self.group_frames
    def __getitem__(self, idx):
        item,action,world_level =  super().__getitem__(idx)
        action = self._extract_action(self.file_names[idx + self.group_frames])
        return item,action,world_level
class MarioHistoryButtonsDataset(MarioButtonsDataset):
    def __init__(self, img_dir, history_size: int = 7, worlds = TRAIN_WORLDS,use_color=False, preload=False):
        super().__init__(img_dir, group_frames=history_size, use_color=use_color, worlds=worlds, preload=preload)
    
    def _load_episodes(self, worlds):
        for file in os.listdir(self.img_dir):
            if not os.path.isdir(os.path.join(self.img_dir,file)):
                continue
            if "win" not in file:
                continue
            mario_episode = MarioHistoryEpisode(os.path.join(self.img_dir,file),self.group_frames,self.use_color,self.transform,preload=False)
            if int(mario_episode.world) in worlds:
                mario_episode = MarioHistoryEpisode(os.path.join(self.img_dir,file),self.group_frames,self.use_color,self.transform,preload=False)
                self.episodes.append(mario_episode)
                self.total_length += len(mario_episode)
        print(f"total episodes:{len(self.episodes)}")

    
    