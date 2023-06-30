import glob
import os
import torch
import torchvision
from PIL import Image
import cv2
from helper_code.resnet_model import AgentModel
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT,COMPLEX_MOVEMENT


env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, COMPLEX_MOVEMENT)

done = False

def image_seq_to_video(imgs_path, output_path='./video.mp4', fps=15.0):
    output = output_path
    img_array = []
    for filename in sorted(glob.glob(os.path.join(imgs_path, '*.jpg')), key=os.path.getmtime):
        img = cv2.imread(filename)
        height, width, layers = img.shape
        # img = cv2.resize(img, (width // 2, height // 2))
        img = cv2.resize(img, (width, height))
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)

    print(size)
    print("writing video...")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Be sure to use lower case
    out = cv2.VideoWriter(output, fourcc, fps, size)
    # out = cv2.VideoWriter('project.avi', cv2.VideoWriter_fourcc(*'DIVX'), 15, size)

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()
    print("saved video @ ", output)
def action_mapper(action_tensor):
    COMPLEX_MOVEMENT_DICT = {
        "0-0-0-0":0,#['NOOP'],
        "0-0-1-0":1,#['right'],
        "1-0-1-0":2,#['right', 'A'],
        "0-0-1-1":3,#['right', 'B'],
        "1-0-1-1":4,#['right', 'A', 'B'],
        "1-0-0-0":5,#['A'],
        "0-1-0-0" : 6,#['left'],
        "1-1-0-0" : 7,#['left', 'A'],
        "0-1-0-1" : 8,#['left', 'B'],
        "1-1-0-1": 9,#['left', 'A', 'B'],
        #['down'],
        #['up'],
    }
    action_tensor = action_tensor[0].cpu()
    key = "-".join([str(int(action_tensor[i].item())) for i in range(4)])
    action = COMPLEX_MOVEMENT_DICT[key]
    return action
#model
history_size = 7
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_path = "./models/agents_model/model_final_agent.pt"
model = AgentModel(history_size=history_size,use_color=False).to(device)
model.load_state_dict(torch.load(model_path))
model.eval()

q_frames_history = []
transform =  torchvision.transforms.Compose([
            torchvision.transforms.Resize((256,256)),
            torchvision.transforms.Grayscale(num_output_channels=1),
            torchvision.transforms.ToTensor(),
            
            #torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

all_frames = []

print(f"action space:{env.action_space}")
env.reset()
for step in range(100):
    if done:
        break
    if len(q_frames_history) < history_size:
        action = 0 #['NOOP']
    else:
        model_input = torch.zeros((7,256,256))
        input_tensor = torch.stack(q_frames_history).unsqueeze(0)
        image = input_tensor.to(device)
        action_tensor = torch.round(torch.sigmoid(model(image)))
        action = action_mapper(action_tensor)

    state, reward, done, info = env.step(action)
    print(f"step:{step} , action:{action} , reward:{reward} , done:{done} , info:{info}")
    pil_img = Image.fromarray(state)
    opencvImage = cv2.cvtColor(state, cv2.COLOR_RGB2BGR)
    cv2.imwrite('../test_run/{step}.jpg',opencvImage)
    q_frames_history.append(transform(pil_img).squeeze())
    q_frames_history.pop(0)

    #env.render()

image_seq_to_video("../test_run/",output_path='./video/mario_play.mp4',fps=30.0)
env.close()