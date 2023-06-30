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
    key = "-".join([action_tensor[i].item() for i in range(4)])
    action = COMPLEX_MOVEMENT_DICT[key]
    return action
#model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_path = "./models/agents_model/model_final_agent.pt"
model = AgentModel(history_size=7,use_color=False).to(device)
model.load_state_dict(torch.load(model_path))
model.eval()

q_frames_history = []
transform =  torchvision.transforms.Compose([
            torchvision.transforms.Resize((256,256)),
            torchvision.transforms.Grayscale(num_output_channels=1),
            torchvision.transforms.ToTensor(),
            
            #torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
out = cv2.VideoWriter('./video/mario_play.mp4', fourcc, 30  ,(256,256))


print(f"action space:{env.action_space}")
env.reset()
for step in range(100):
    if done:
        break
    if len(q_frames_history) < 7:
        action = env.action_space.sample() #['NOOP']
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
    q_frames_history.append(transform(pil_img).squeeze())
    out.write(opencvImage)
    #env.render()

env.close()