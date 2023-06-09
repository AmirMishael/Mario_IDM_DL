import pathlib
from helper_code.resnet_model import ResnetModel
from helper_code.mario_buttons_dataset import TEST_WORLDS, TRAIN_WORLDS, VAL_WORLDS, MarioButtonsDataset
import torch
import torchvision
import os
from PIL import Image
from tqdm import tqdm

def calculate_accuracy(model, dataloader, device):
    print("calculating accuracy")
    model.eval() # put in evaluation mode,  turn of DropOut, BatchNorm uses learned statistics
    total_correct = 0
    total_inputs = 0
    #confusion_matrix = np.zeros([10,10], int)
    with torch.no_grad():
        for data in tqdm(dataloader,desc="calculating accuracy"):
            inputs, buttons ,world_level = data
            inputs = inputs.to(device)
            buttons = buttons.to(device)
            outputs = model(inputs)
            predicted = torch.round(torch.sigmoid(outputs))
            
            #print(f"size of predicted: {buttons.size()}")
            total_inputs += buttons.size(0) * buttons.size(1)
            total_correct += (predicted == buttons).sum().item()

    model_accuracy = total_correct / total_inputs
    print(f"accuracy: {model_accuracy}")
    return model_accuracy 

def train_loop(model,data_loader,val_loader,device,group,epochs,learning_rate,save_path='./models',aug_list=[],start_epoch=0,start_batch=0):
    print(f"started training with hyperparams: group:{group}, epochs:{epochs}, learning_rate:{learning_rate}")
    loss_history=[]
    max_val_accuracy = 0

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    #optimizer = torch.optim.Adam(model.resnet.fc.parameters(), lr=learning_rate)
    criterion = torch.nn.BCEWithLogitsLoss()
    
    # for _ in range(start_batch):
    #     next(data_loader)

    for epoch in range(start_epoch+1,epochs):
        model.train()
        running_loss = 0.0
        for i,data in enumerate(tqdm(data_loader,desc=f"training epoch:{epoch}")):
            inputs,buttons,world_level = data
            inputs = inputs.to(device)
            buttons = buttons.to(device)

            output = model(inputs)
            loss = criterion(output,buttons)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            los_val = loss.data.item()
            running_loss += los_val

            if i % 500 == 0:
                print(f"saving checkpoint at epoch:{epoch}, batch:{i}, loss:{los_val}")
                torch.save(model.state_dict(),f"{save_path}/checkpoints/checkpoint_{epoch}_{i}.pt")
    
        loss_history.append(running_loss)
        running_loss /= len(data_loader)

        val_accuracy = calculate_accuracy(model,val_loader,device)
        if val_accuracy > max_val_accuracy:
            max_val_accuracy = val_accuracy
            torch.save(model.state_dict(),f"{save_path}/best_model.pt")
        print(f"running loss : {running_loss} , epoch:{epoch}")

def main_train(models_dir = "./models",checkpoint_path=None):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    torch.manual_seed(17)


    batch_size = 128
    learning_rate = 1e-3
    epochs = 7
    
    group = 3#3
    use_color = False

    preload=False

    start_epoch = 0
    start_batch = 0

    
    print(f"loading dataset preload:{preload}")
    mario_dataset_train = MarioButtonsDataset(img_dir='./mario_dataset',group_frames=group,use_color=use_color,worlds=TRAIN_WORLDS,preload=preload)
    mario_dataset_test = MarioButtonsDataset(img_dir='./mario_dataset',group_frames=group,use_color=use_color,worlds=TEST_WORLDS,preload=preload)
    mario_dataset_val = MarioButtonsDataset(img_dir='./mario_dataset',group_frames=group,use_color=use_color,worlds=VAL_WORLDS,preload=preload)
    print(f"tot train dataset frames :{len(mario_dataset_train)}")

    train_loader = torch.utils.data.DataLoader(mario_dataset_train,batch_size=batch_size,shuffle=True,num_workers=8)
    test_loader = torch.utils.data.DataLoader(mario_dataset_test,batch_size=batch_size,shuffle=True,num_workers=8)
    val_loader = torch.utils.data.DataLoader(mario_dataset_val,batch_size=batch_size,shuffle=True,num_workers=8)

    model = ResnetModel(group_size=group,use_color=use_color,use_pretrained=True).to(device)
    if checkpoint_path:
        model.load_state_dict(torch.load(checkpoint_path))
        start_epoch = int(pathlib.PurePath(checkpoint_path).name.split('_')[1])
        start_batch = int(pathlib.PurePath(checkpoint_path).name.split('_')[2].split('.')[0])
    
    
    train_loop(model = model,
          data_loader = train_loader,
          val_loader = val_loader,
          device=device,
          group=group,
          epochs=epochs,
          learning_rate=learning_rate,
          save_path=models_dir,
          start_epoch=start_epoch,
          start_batch=start_batch)
    torch.save(model.state_dict(),f"{models_dir}/model_final.pt")
    print("model saved")