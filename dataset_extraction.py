import glob
import torch
import torchvision
import cv2
import os

from torch.distributed import group
from tqdm import tqdm
import pandas as pd
from PIL import Image
from helper_code.mario_buttons_dataset import MarioEpisode
from helper_code.resnet_model import ResnetModel
from pathlib import Path

## params
video_path = './video/video.mp4'
frames_dir = './video/frames'
start_sec = 10
stop_sec = 35*60
metadata_path = './video/metadata.csv'
group_size = 15

model_path = './models/best_model_group_15_color_False.pt'

## end params

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(17)
transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((256,256)),
            torchvision.transforms.Grayscale(num_output_channels=1),
            torchvision.transforms.ToTensor(),

            #torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])
def extract_frames(video_path, frames_dir,start_sec,stop_sec):
    print("extracting frames")
    vidcap = cv2.VideoCapture(video_path)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    start_frame = int(start_sec * fps)
    stop_frame = int(stop_sec * fps)
    tot_frames = stop_frame - start_frame
    vidcap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    for i in tqdm(range(tot_frames)):
        success, image = vidcap.read()
        image = cv2.resize(image, (256, 256))
        cv2.imwrite(os.path.join(frames_dir, "{:d}.jpg".format(i)), image)

def create_metadata(frames_dir,metadata_path,model,group_size):
    print("creating metadata")
    tensor_indexes = [0,2,3,5] # up, left, right, B,
    df = pd.DataFrame(columns=['id,image_path','up','left','right','B'])
    model.eval()
    files = sorted(glob.glob(os.path.join(frames_dir, '*.jpg')), key=lambda x: int(Path(x).name.split('.')[0]))
    #group every 15 frames
    q = [transform(Image.open(filename)).squeeze() for filename in files[:group_size]]
    for i,filename in tqdm(enumerate(files[group_size:])):
        input_tensor = torch.stack(q).unsqueeze(0)
        image = input_tensor.to(device)
        label_tensor = torch.sigmoid(model(image))
        label_item = label_tensor[0]
        real_file_name = files[i+group_size//2]
        row = pd.DataFrame({'id':[Path(real_file_name).name.split('.')[0]]
                        ,'image_path':[real_file_name]
                        ,'up':[label_item[0].item()],
                        'left':[label_item[1].item()],
                        'right':[label_item[2].item()],
                        'B':[label_item[3].item()]})

        df = pd.concat([df,row],ignore_index=True)
        q.pop(0)
        q.append(transform(Image.open(files[i+group_size])).squeeze())
    df.to_csv(metadata_path,index=False)

def convert_buttons_to_history(episode_path:str,history_size:int = 7 , metadata_path:str = './video/metadata.csv',frames_path:str = './video/frames'):
    mario_buttons_ep = MarioEpisode(episode_dir=episode_path,group_frames=history_size,transform=None)
    metadata_df = pd.DataFrame(columns=['id,image_path','up','left','right','B'])
    if not os.path.exists(frames_path):
        os.makedirs(frames_path)
    for i in range(len(mario_buttons_ep)):
        img = mario_buttons_ep._get_image(i)
        img.save(os.path.join(frames_path,f"{i}.jpg"))
        if i >= history_size:
            real_file_name = os.path.join(frames_path,f"{i}.jpg")
            action = mario_buttons_ep._extract_action(os.path.join(episode_path,f"{mario_buttons_ep.file_names[i]}.jpg"))
            row = pd.DataFrame({'id':[Path(real_file_name).name.split('.')[0]]
                            ,'image_path':[real_file_name]
                            ,'up':[action[0].item()],
                            'left':[action[1].item()],
                            'right':[action[2].item()],
                            'B':[action[3].item()]})

            metadata_df = pd.concat([metadata_df,row],ignore_index=True)
    metadata_df.to_csv(metadata_path,index=False) 
        
        
extract_frames(video_path=video_path, frames_dir= frames_dir,start_sec=start_sec,stop_sec=stop_sec)
model = ResnetModel(group_size=group_size,use_color=False).to(device)
model.load_state_dict(torch.load(model_path))
create_metadata(frames_dir=frames_dir,metadata_path=metadata_path,model=model,group_size=group_size)
# for file in os.listdir('./mario_dataset/'):
#     if "win" in file:
#         convert_buttons_to_history(episode_path=f'./mario_dataset/{file}',history_size=7,metadata_path=f'./video/converted/metadata_{file}.csv',frames_path=f'./video/converted/{file}_frames')

