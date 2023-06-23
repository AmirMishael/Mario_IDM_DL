import glob
import torch
import torchvision
import cv2
import os
from tqdm import tqdm
import pandas as pd
from PIL import Image

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(17)
transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((256,256)),
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

def create_metadata(frames_dir,metadata_path,model):
    print("creating metadata")
    tensor_indexes = [0,2,3,5] # up, left, right, B,
    df = pd.DataFrame(columns=['id,image_path','up','left','right','B'])
    model.eval()
    for filename in tqdm(sorted(glob.glob(os.path.join(frames_dir, '*.jpg')), key=int(filename.split('.')[0]))):
        image = Image.open(filename)
        image = transform(image).to(device)
        label_tensor = model(image)
        label_item = label_tensor[0]
        df = df.append({'id':filename.split('.')[0]
                        ,'image_path':filename
                        ,'up':label_item[0].item(),
                        'left':label_item[1].item(),
                        'right':label_item[2].item(),
                        'B':label_item[3].item()},ignore_index=True)
    df.to_csv(metadata_path,index=False)

    
extract_frames('./video.mp4', './frames',start_sec=10,stop_sec=35*60)
model = torch.load('./models/checkpoint_1_3500_group_15_color_False.pt').to(device)
create_metadata(frames_dir='./frames',metadata_path='./metadata.csv',model=model)
