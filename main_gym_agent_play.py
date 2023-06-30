import torch
import torchvision
from helper_code.resnet_model import AgentModel
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT,COMPLEX_MOVEMENT
env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, COMPLEX_MOVEMENT)

done = False


def action_mapper(action_tensor):
    action_tensor = action_tensor[0].cpu()
    action = []
    if action_tensor[1].item() > 0.5:
        action.append("left")
    if action_tensor[2].item() > 0.5:
        action.append("right")
    if action_tensor[0].item() > 0.5:
        action.append("up")
    if action_tensor[3].item() > 0.5:
        action.append("B")
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
            #torchvision.transforms.Grayscale(num_output_channels=1),
            torchvision.transforms.ToTensor(),
            
            #torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

env.reset()
for step in range(10):
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
    print(f"state:{state.shape}")
    #env.render()

env.close()