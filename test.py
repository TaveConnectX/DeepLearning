import torch
import numpy as np
import math
from env import ConnectFour, MCTS_alphago, Node_alphago
from models import ResNetforDQN
import copy
import torch.nn as nn


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(42, 84)  # 입력 크기: 42, 출력 크기: 임의로 설정한 중간 층 크기
        self.fc2 = nn.Linear(84, 3)  # 입력 크기: 중간 층 크기, 출력 크기: 클래스 수
        self.relu = nn.ReLU()

    def forward(self, x):
        # x = x.view(x.size(0), -1)  # 2차원 배열을 1차원으로 평탄화
        x = x.flatten()  # 일반적인 사용을 위해 수정
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
CF = ConnectFour()
args = {
    'C': 1,
    'num_searches': 100,
    'dirichlet_epsilon': 0.25,
    'dirichlet_alpha': 1,
    'temperature':2
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNetforDQN(action_size=49)
model.load_state_dict(torch.load("model/models_for_V_net/model_RL/Model81_DQN-resnet-minimax-v1.pth",  map_location=device))
model.eval()

value_model = Classifier().to(device)
value_model.load_state_dict(torch.load("model/models_for_V_net/ValueNetwork.pth", map_location=device))

mcts = MCTS_alphago(CF, args, model,value_model=value_model)

state = CF.get_initial_state()
# state = torch.from_numpy(state).float().unsqueeze(0).unsqueeze(0)
# state = state.to(device)
# print(model(state))

# action_probs = mcts.search(state)
# print(action_probs)
player = np.random.choice([1,-1])
# player = -1




def print_state(board):
    X_mark = "\033[31mX\033[0m"
    O_mark = "\033[33mO\033[0m"
    n_row, n_col = 6,7
    print("Connect Four")
    print("-----------------------")
    empty_space = [" "]*n_col
    board = copy.deepcopy(board)
    
    
    row_str = " "
    for col in range(n_col):
        row_str += empty_space[col]
        row_str += " "
    print(row_str)
    for row in range(n_row):
        row_str = "|"
        for col in range(n_col):
            if board[row][col] == 0:
                row_str += " "
            elif board[row][col] == 1:
                row_str += X_mark
            elif board[row][col] == -1:
                row_str += O_mark
            row_str += "|"
        print(row_str)
    print("+" + "-" * (len(board[0]) * 2 - 1) + "+")
    print("player {}'s turn!".format(int(player)))

while True:
    print_state(state)
    if player == 1:
        # print("state", state)
        # print("state", state[0][0].detach().cpu().numpy())
        valid_moves = CF.get_valid_moves(state)
        print("valid_moves", [i for i in range(CF.action_size) if valid_moves[i] == 1])
        action = int(input(f"{player}:"))

        if valid_moves[action] == 0:
            print("action not valid")
            continue
        
        
            
    else:
        neutral_state = CF.change_perspective(state, player)
        mcts_probs = mcts.search(neutral_state)
        print("player -1:",mcts_probs)
        # time.sleep(5)
        action = np.argmax(mcts_probs)
        # action = np.random.choice(range(7),p=mcts_probs)  
    state = CF.get_next_state(state, action, player)
    
    value, is_terminal = CF.get_value_and_terminated(state, action)
    
    if is_terminal:
        print(state)
        if value == 1:
            print(player, "won")
        else:
            print("draw")
        break
        
    player = CF.get_opponent(player)


print_state(state)