import pathlib
from helper_code.mario_history_dataset import MarioHistoryButtonsDataset, MarioHistoryDataset
from helper_code.resnet_model import AgentModel, ResnetModel
from helper_code.mario_buttons_dataset import TEST_WORLDS, TRAIN_TEST, TRAIN_WORLDS, VAL_WORLDS, MarioButtonsDataset
import torch
import torchvision
import os
import kornia.augmentation as K
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
    print(f"accuracy: {model_accuracy} , total_correct:{total_correct} , total_inputs:{total_inputs}")
    return model_accuracy 

def train_loop(model,data_loader,val_loader,device,group,epochs,learning_rate,use_color,save_path='./models'
               ,aug_list=[],start_epoch=0,pos_weight=None):
    print(f"started training with hyperparams: group:{group}, epochs:{epochs}, learning_rate:{learning_rate} ,use_color:{use_color} ,aug_list:{len(aug_list)},start_epoch:{start_epoch}")
    loss_history=[]
    max_val_accuracy = 0

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    #optimizer = torch.optim.Adam(model.resnet.fc.parameters(), lr=learning_rate)
    criterion = torch.nn.BCEWithLogitsLoss()
    if pos_weight is not None:
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # for _ in range(start_batch):
    #     next(data_loader)
    for  i in range(start_epoch):
        scheduler.step()
    for epoch in range(start_epoch+1,epochs):
        model.train()
        running_loss = 0.0
        for i,data in enumerate(tqdm(data_loader,desc=f"training epoch:{epoch}")):
            inputs,buttons,world_level = data
            inputs = inputs.to(device)
            buttons = buttons.to(device)
            if aug_list:
                inputs = aug_list[torch.randint(0,len(aug_list),(1,))](inputs)
            output = model(inputs)
            loss = criterion(output,buttons)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            los_val = loss.data.item()
            running_loss += los_val

            if i % 500 == 0:
                print(f"saving checkpoint at epoch:{epoch}, batch:{i}, loss:{los_val}")
                torch.save(model.state_dict(),f"{save_path}/checkpoints/checkpoint_{epoch}_{i}_group_{group}_color_{use_color}.pt")
        
        loss_history.append(running_loss)
        running_loss /= len(data_loader)

        val_accuracy = calculate_accuracy(model,val_loader,device)
        if val_accuracy > max_val_accuracy:
            max_val_accuracy = val_accuracy
            torch.save(model.state_dict(),f"{save_path}/best_model_group_{group}_color_{use_color}.pt")
        print(f"running loss : {running_loss} , epoch:{epoch} ,max_val_accuracy:{max_val_accuracy} ,val_accuracy:{val_accuracy}")
        scheduler.step()

def main_train_agent(models_dir = "./models",start_epoch=0,lr=1e-3,group=7,use_color=False,use_aug=False):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    torch.manual_seed(17)


    batch_size = 128
    learning_rate = lr
    epochs = 10
    
    group = group
    use_color = use_color

    preload=False
    if use_aug:
        aug_ls = [K.RandomGaussianNoise(mean=0,std=0.05,p=0.2),
                  K.RandomInvert(p=0.2),
                  K.RandomBoxBlur(kernel_size=(3,3),p=0.2),
                  K.RandomErasing(p=0.2,scale=(0.02,0.05)),
                  K.RandomRotation(degrees=5,p=0.2)]
    else:
        aug_ls = []


    
    print(f"loading dataset preload:{preload}")
    # mario_dataset = MarioHistoryDataset(img_dir='./video/frames',history_frames=group,use_color=use_color,preload=preload,metadata_file='./video/metadata.csv' )
    # mario_dataset_train,mario_dataset_test,mario_dataset_val = torch.utils.data.random_split(mario_dataset,[0.89,0.01,0.1])
    mario_dataset_train = MarioHistoryButtonsDataset(history_size=group,use_color=False,img_dir="./mario_dataset",preload=preload,worlds=TRAIN_TEST)
    mario_dataset_val = MarioHistoryButtonsDataset(history_size=group,use_color=False,img_dir="./mario_dataset",preload=preload,worlds=VAL_WORLDS)
    print(f"tot train dataset frames :{len(mario_dataset_train)}")

    train_loader = torch.utils.data.DataLoader(mario_dataset_train,batch_size=batch_size,shuffle=True,num_workers=4)
    #test_loader = torch.utils.data.DataLoader(mario_dataset_test,batch_size=batch_size,shuffle=True,num_workers=8)
    val_loader = torch.utils.data.DataLoader(mario_dataset_val,batch_size=batch_size,shuffle=True,num_workers=4)
    
    # additional_loaders = []
    # for file_name in os.listdir('./video/converted'):
    #     if ".csv" in file_name:
    #         name = file_name.split(".")[0].replace("metadata_","")
    #         additional_dataset = MarioHistoryDataset(img_dir=f'./video/converted/{name}_frames',history_frames=group,use_color=use_color,preload=preload,metadata_file=f'./video/converted/metadata_{name}.csv' )
    #         additional_loader = torch.utils.data.DataLoader(additional_dataset,batch_size=batch_size,shuffle=True,num_workers=4)
    #         additional_loaders.append(additional_loader)
    #additional_loaders.append(train_loader)

    model = AgentModel(history_size=group,use_color=use_color).to(device)    
    if start_epoch > 0:
        model.load_state_dict(torch.load(f"{models_dir}/best_model_group_{group}_color_{use_color}.pt"))
    
    train_loop(model = model,
          data_loader = train_loader,
          val_loader = val_loader,
          device=device,
          group=group,
          epochs=epochs,
          learning_rate=learning_rate,
          use_color=use_color,
          save_path=models_dir,
          aug_list=aug_ls,
          start_epoch=start_epoch,
          )
    torch.save(model.state_dict(),f"{models_dir}/model_final_agent.pt")
    print("model saved")

def main_train(models_dir = "./models",checkpoint_path=None,start_epoch=0,lr=1e-3,group=7,use_color=False,use_aug=False):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    torch.manual_seed(17)


    batch_size = 128
    learning_rate = lr
    epochs = 7
    
    group = group
    use_color = use_color

    preload=False
    if use_aug:
        aug_ls = [K.RandomGaussianNoise(mean=0,std=0.05,p=0.2),
                  K.RandomInvert(p=0.2),
                  K.RandomBoxBlur(kernel_size=(3,3),p=0.2),
                  K.RandomErasing(p=0.2,scale=(0.02,0.05)),
                  K.RandromRotation(degrees=5,p=0.2)]
    else:
        aug_ls = []


    
    print(f"loading dataset preload:{preload}")
    mario_dataset_train = MarioButtonsDataset(img_dir='./mario_dataset',group_frames=group,use_color=use_color,worlds=TRAIN_WORLDS,preload=preload)
    mario_dataset_test = MarioButtonsDataset(img_dir='./mario_dataset',group_frames=group,use_color=use_color,worlds=TEST_WORLDS,preload=preload)
    mario_dataset_val = MarioButtonsDataset(img_dir='./mario_dataset',group_frames=group,use_color=use_color,worlds=VAL_WORLDS,preload=preload)
    print(f"tot train dataset frames :{len(mario_dataset_train)}")

    train_loader = torch.utils.data.DataLoader(mario_dataset_train,batch_size=batch_size,shuffle=True,num_workers=4)
    #test_loader = torch.utils.data.DataLoader(mario_dataset_test,batch_size=batch_size,shuffle=True,num_workers=8)
    val_loader = torch.utils.data.DataLoader(mario_dataset_val,batch_size=batch_size,shuffle=True,num_workers=4)

    model = ResnetModel(group_size=group,use_color=use_color,use_pretrained=True).to(device)
    if checkpoint_path:
        model.load_state_dict(torch.load(checkpoint_path))
        #start_batch = int(pathlib.PurePath(checkpoint_path).name.split('_')[2].split('.')[0])
    
    
    train_loop(model = model,
          data_loader = train_loader,
          val_loader = val_loader,
          device=device,
          group=group,
          epochs=epochs,
          learning_rate=learning_rate,
          use_color=use_color,
          save_path=models_dir,
          aug_list=aug_ls,
          start_epoch=start_epoch,)
    torch.save(model.state_dict(),f"{models_dir}/model_final.pt")
    print("model saved")
