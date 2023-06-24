import torch
import torchvision
import os
from torch.utils.data import Dataset
from PIL import Image
from tqdm import tqdm
import pathlib
from helper_code.eval_model import calc_accuracy_agent, hist_correct_world_level,calc_accuracy
from helper_code.resnet_model import AgentModel, ResnetModel

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(17)

modles_path = pathlib.Path("./models/agents_model")

history_size = 7
use_color = False
models_to_check = ["checkpoints/checkpoint_1_3500_group_15_color_False.pt","checkpoints/checkpoint_2_3500_group_15_color_False.pt",
                   "checkpoints/checkpoint_3_3500_group_15_color_False.pt"
                   ,"checkpoints/checkpoint_4_3500_group_15_color_False.pt","checkpoints/checkpoint_5_3500_group_15_color_False.pt"
                   ,"checkpoints/checkpoint_6_3500_group_15_color_False.pt"]
for model_name in models_to_check:
    print(f"checking model: {model_name}")
    model = AgentModel(history_size=history_size,use_color=use_color,use_pretrained=False).to(device)
    model.load_state_dict(torch.load(os.path.join(modles_path,model_name)))
    print("val")
    calc_accuracy_agent(model,mode='val',history=history_size,use_color=use_color)
    print("test")
    calc_accuracy_agent(model,mode='test',history=history_size,use_color=use_color)
