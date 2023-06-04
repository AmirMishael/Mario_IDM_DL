import torch
import torchvision
import os
from torch.utils.data import Dataset
from PIL import Image
from tqdm import tqdm
import pathlib


class MarioEpisode(Dataset):
    #format of folder <user>_<sessid>_e<episode>_<world>-<level>_<outcome>
    def __init__(self,episode_dir,group_frames:int=1,transform = None,preload = False):
        super().__init__()
        self.episode_dir = episode_dir
        self.group_frames = group_frames
        self.transform = transform

        metadata = pathlib.PurePath(episode_dir).name.split("_")
        self.user = metadata[0]
        self.sessid = metadata[1]
        self.episode = metadata[2][1:]
        #print(metadata)
        self.world = metadata[3].split("-")[0]
        self.level = metadata[3].split("-")[1]
        self.outcome = metadata[4]

        #using medatata format: <user>_<sessid>_e<episode>_<world>-<level>_f<frame>_a<action>_<datetime>.<outcome>.png
        
        self.file_names = os.listdir(episode_dir)
        self.file_names.sort(key=lambda dir : int(pathlib.PurePath(dir).name.split("_")[4][1:]))
        self.total_frames = len(self.file_names)

        self.preloaded_images = None

        self.shape = (256,256)#Image.open(os.path.join(episode_dir,self.file_names[0])).convert("L").size
        if preload:
            self.preloaded_images = []
            for file in tqdm(self.file_names,desc=f"preloading {episode_dir}"):
                current_img = Image.open(os.path.join(self.episode_dir,file)).convert("L")
                if self.transform:
                    current_img = self.transform(current_img)
                self.preloaded_images.append(current_img)

        
    
    def __len__(self):
        return self.total_frames - self.group_frames + 1
    def _get_image(self,idx):
        if self.preloaded_images is not None:
            return self.preloaded_images[idx]
        else:
            current_img = Image.open(os.path.join(self.episode_dir,self.file_names[idx])).convert("L")
            if self.transform:
                current_img = self.transform(current_img)
            return current_img
    def __getitem__(self, idx) :
        item = torch.zeros((self.group_frames,self.shape[1],self.shape[0]))

        for i in range(self.group_frames):
            current_img = self._get_image(idx+i)
            item[i] = current_img
        return item, self._extract_action(self.file_names[idx+int(self.group_frames/2)]) #action from mid frame
    #using medatata format: <user>_<sessid>_e<episode>_<world>-<level>_f<frame>_a<action>_<datetime>.<outcome>.png
    #action format is move to binary and from msb to lsb : up, down, left, right, A, B, start, select, e.g.: 20dec = 00010100bin = right + B (running to the right)
    def _extract_action(self, file_name):
        #print(pathlib.PurePath(file_name).name)
        action_number = pathlib.PurePath(file_name).name.split("_")[5][1:]
        action_tensor = torch.zeros(8)
        action_bin = str(bin(int(action_number)))[2:].zfill(8) #remove 0b in start
        #print(action_bin)
        for i in range(8):
            action_tensor[i] = int(action_bin[i])
        return action_tensor
            
                   
        
class MarioButtonsDataset(Dataset):
    def __init__(self,img_dir,group_frames:int = 1, transform = None,preload = False):
        super().__init__()
        self.group_frames = group_frames
        self.img_dir = img_dir
        self.episodes = []
        
        self.total_length = 0
        for file in os.listdir(img_dir):
            mario_episode = MarioEpisode(os.path.join(img_dir,file),group_frames,transform,preload)
            self.episodes.append(mario_episode)
            self.total_length += len(mario_episode)
        print(f"total episodes:{len(self.episodes)}")
           
    def __len__(self):
        return self.total_length
    def __getitem__(self, idx):
        current_length = 0
        for episode in self.episodes:
            current_length += len(episode)
            #print(f"current:{current_length}, tot:{self.total_length},episode:{episode.total_frames}")
            if idx < current_length:
                return episode[idx % len(episode)]
        raise IndexError(f"Index out of range. index:{idx} ,tot:{self.total_length}")
