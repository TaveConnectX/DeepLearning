from functions import get_model_and_config_name, board_normalization, \
                        load_model
import env
from agent_structure import ConnectFourDQNAgent
from models import *
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import copy

device = "cuda:0" if torch.cuda.is_available() else "cpu"

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(42, 84)  # 입력 크기: 42, 출력 크기: 임의로 설정한 중간 층 크기
        self.fc2 = nn.Linear(84, 3)  # 입력 크기: 중간 층 크기, 출력 크기: 클래스 수
        self.relu = nn.ReLU()

    def forward(self, x):
        # x = x.view(x.size(0), -1)  # 2차원 배열을 1차원으로 평탄화
        x = x.flatten()
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
def board_normalization(state, player):
    state_ = copy.deepcopy(state)
    state_ = np.array(state_)
    state_[state_==2] = -1

    state_ *= player
    return torch.FloatTensor(state_).to(device)

model = Classifier().to(device)
model.load_state_dict(torch.load("model/models_for_V_net/ValueNetwork.pth", map_location=device))

state = [[
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 2, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0],
    [0, 0, 1, 1, 0, 0, 0],
    [0, 0, 2, 1, 2, 0, 0],
    [0, 0, 2, 2, 1, 0, 0]
]]

norm_state = board_normalization(state,player=1)

print("state:\n", np.array(state))
print("normalized state:\n",norm_state)

output = model(norm_state)
print(output)
prob = nn.functional.softmax(output)
print(prob)

