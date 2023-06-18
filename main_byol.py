import os
import pathlib
import torch
import kornia.augmentation as K
from helper_code.my_byol import BYOL
from helper_code.resnet_model import ResnetModel
from helper_code.mario_buttons_dataset import TRAIN_WORLDS, VAL_WORLDS, MarioButtonsDataset


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(17)

modles_path = pathlib.Path("./models")
model_name = f""

group = 15
use_color = False
batch_size = 128

model = ResnetModel(group_size=group,use_color=use_color,use_pretrained=False).to(device)
model.load_state_dict(torch.load(os.path.join(modles_path,model_name)))

mario_dataset_train = MarioButtonsDataset(img_dir='./mario_dataset',group_frames=group,use_color=use_color,worlds=TRAIN_WORLDS,preload=False)
mario_dataset_val = MarioButtonsDataset(img_dir='./mario_dataset',group_frames=group,use_color=use_color,worlds=VAL_WORLDS,preload=False)
print(f"tot train dataset frames :{len(mario_dataset_train)}")

train_loader = torch.utils.data.DataLoader(mario_dataset_train,batch_size=batch_size,shuffle=True,num_workers=4)
#test_loader = torch.utils.data.DataLoader(mario_dataset_test,batch_size=batch_size,shuffle=True,num_workers=8)
val_loader = torch.utils.data.DataLoader(mario_dataset_val,batch_size=batch_size,shuffle=True,num_workers=4)



learner = BYOL(
    model,
    channels=group*3 if use_color else group,
    image_size = 256,
    hidden_layer = 'avgpool',
    augment_fn=K.AugmentationSequential(K.RandomGaussianBlur(p=0.5),
                                        K.RandromRotation(degrees=5,p=0.5)),
    augment_fn2=K.AugmentationSequential(K.RandomResizedCrop(p=0.5,size=(256,256)),
                                         K.RandomInvert(p=0.5)),

)


opt = torch.optim.Adam(learner.parameters(), lr=3e-4)

def sample_unlabelled_images():
    images, _ = next(iter(train_loader))
    return images

for _ in range(10000):
    images = sample_unlabelled_images()
    loss = learner(images)
    opt.zero_grad()
    loss.backward()
    opt.step()
    learner.update_moving_average() # update moving average of target encoder

# save your improved network
torch.save(model.state_dict(), os.path.join(modles_path,'improved-net.pt'))