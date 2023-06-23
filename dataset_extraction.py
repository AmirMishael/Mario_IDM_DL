import glob
import torch
import torchvision
import cv2
import os
from tqdm import tqdm
import pandas as pd
from PIL import Image
from helper_code.resnet_model import ResnetModel
from pathlib import Path

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
    for i,filename in tqdm(enumerate(files[:len(files)-group_size])):
        input_tensor = torch.stack(q).unsqueeze(0)
        image = input_tensor.to(device)
        label_tensor = model(image)
        label_item = label_tensor[0]
        real_file_name = files[i+group_size//2]
        row = pd.DataFrame({'id':[Path(real_file_name).name.split('.')[0]]
                        ,'image_path':[filename]
                        ,'up':[label_item[0].item()],
                        'left':[label_item[1].item()],
                        'right':[label_item[2].item()],
                        'B':[label_item[3].item()]})

        df = pd.concat([df,row],ignore_index=True)
        q.pop(0)
        q.append(transform(Image.open(files[i+group_size])).squeeze())
    df.to_csv(metadata_path,index=False)


#extract_frames('./video/video.mp4', './video/frames',start_sec=10,stop_sec=35*60)
group_size = 15
model = ResnetModel(group_size=group_size,use_color=False).to(device)
model.load_state_dict(torch.load('./models/best_model_group_15_color_False.pt'))
create_metadata(frames_dir='./video/frames',metadata_path='./video/metadata.csv',model=model,group_size=group_size)