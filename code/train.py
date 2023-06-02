from resnet_model import ResnetModel
from mario_buttons_dataset import MarioButtonsDataset
import torch
import torchvision
import os
from PIL import Image
from tqdm import tqdm


def train(model,data_loader,val_loader,device,group,epochs,learning_rate,save_path='./models',aug_list=[]):
    loss_history=[]
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.BCEWithLogitsLoss()
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for i,data in tqdm(enumerate(data_loader)):
            inputs,buttons = data
            inputs = inputs.to(device)
            buttons = buttons.to(device)

            output = model(inputs)
            loss = criterion(output,buttons)
            optimizer.zero_grad()
            optimizer.step()

    runnin_loss += loss.data.item()
    loss_history.append(running_loss)
    running_loss /= len(data_loader)
    print(f"running loss : {running_loss} , epoch:{epoch}")

def main_train():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    torch.manual_seed(17)


    batch_size = 128
    learning_rate = 1e-4
    epochs = 7
    group = 3

    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((256,256)),
        torchvision.transforms.ToTensor()
        #transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    
    mario_dataset = MarioButtonsDataset(img_dir='./mario_dataset',group_frames=group,transform=transforms)
    print(f"tot dataset frames :{len(mario_dataset)}")
    train,test,val = torch.utils.data.random_split(mario_dataset,[0.7,0.2,0.1])

    train_loader = torch.utils.data.DataLoader(train,batch_size=batch_size,shuffle=True)
    test_loader = torch.utils.data.DataLoader(test,batch_size=batch_size,shuffle=True)
    val_loader = torch.utils.data.DataLoader(val,batch_size=batch_size,shuffle=True)


    model = ResnetModel(group_size=group).to(device)
    
    train(model,train_loader,val_loader,device,group,epochs,learning_rate)
    torch.save(model.state_dict(),f"./models/model_{group}.pt")
    print("model saved")