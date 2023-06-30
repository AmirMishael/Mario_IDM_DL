import torch
import torchvision
import os
from PIL import Image
from tqdm import tqdm
import pathlib
from helper_code.train import main_train_agent

modles_path = './models/agents_model'

main_train_agent(models_dir=modles_path
           ,start_epoch=0
           ,lr=1e-3
           ,group=7
           ,use_color=False
           ,use_aug=True)