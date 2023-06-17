import pathlib
from helper_code.resnet_model import ResnetModel
from helper_code.mario_buttons_dataset import TEST_WORLDS, TRAIN_WORLDS, VAL_WORLDS, MarioButtonsDataset
import torch
import torchvision
import os
from PIL import Image
from tqdm import tqdm

from helper_code.train import calculate_accuracy

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(17)

def hist_correct_world_level(model):
    print("calculating world level hist")

    mario_dataset_test = MarioButtonsDataset(img_dir='./mario_dataset',group_frames=3,use_color=False,worlds=TEST_WORLDS,preload=False)
    data_loader = torch.utils.data.DataLoader(mario_dataset_test,batch_size=128,shuffle=True,num_workers=8)
    model.eval() # put in evaluation mode,  turn of DropOut, BatchNorm uses learned statistics
    
    world_level_hist = {}
    with torch.no_grad():
        for data in tqdm(data_loader,desc="calculating accuracy"):
            inputs, buttons , world_level = data
            inputs = inputs.to(device)
            buttons = buttons.to(device)
            outputs = model(inputs)
            predicted = torch.round(torch.sigmoid(outputs))
            correct_labels = (predicted == buttons).cpu().numpy()
            world_level = world_level.cpu().numpy()
            for i in range(correct_labels.shape[0]):
                if world_level[i] not in world_level_hist:
                    world_level_hist[world_level[i]] = 0
                world_level_hist[world_level[i]] += correct_labels[i].sum()
            #print(f"size of predicted: {buttons.size()}")
    print(f"world-level correct hist :{world_level_hist}")
    return world_level_hist

def calc_accuracy(model,mode='test',group_frames=7,use_color=False):
    if mode == 'test':
        mario_dataset = MarioButtonsDataset(img_dir='./mario_dataset',group_frames=group_frames,use_color=False,worlds=TEST_WORLDS,preload=False)
    else:
        mario_dataset = MarioButtonsDataset(img_dir='./mario_dataset',group_frames=group_frames,use_color=False,worlds=VAL_WORLDS,preload=False)

    data_loader = torch.utils.data.DataLoader(mario_dataset,batch_size=128,shuffle=True,num_workers=8)
    model.eval() # put in evaluation mode,  turn of DropOut, BatchNorm uses learned statistics
    calculate_accuracy(model,data_loader,device)


