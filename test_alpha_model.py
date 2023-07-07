from env import ConnectFour, MCTS
from models import AlphaZeroResNet
import numpy as np
import torch
import time
import os
import copy
import random
import keyboard
from agent_structure import ConnectFourDQNAgent, MinimaxAgent
from functions import get_model_config, get_model_and_config_name,\
                    board_normalization
# from alphazero_new import ConnectFour, ResNet, MCTS
# import lovely_tensors as lt
# lt.monkey_patch()

X_mark = "\033[31mX\033[0m"
O_mark = "\033[33mO\033[0m"
do_I_play = False
greedy = True
temperature = None if greedy else 0.1


nb1, hl1, model1_name = 9,128,'model_18/model_18_iter_7.pth'
args = {
    'C': 1,
    'num_searches': 350,
    'dirichlet_epsilon': 0.,
    'dirichlet_alpha': 0.3
}

nb2, hl2, model2_name = 9,128,'model_19/model_19_iter_7.pth'
args2 = {
        'C': 1,
        'num_searches': 350,
        'dirichlet_epsilon': 0.,
        'dirichlet_alpha': 0.3
}

# 기준이 되는 모델은 player=1, 상대(비교 모델 or 사람)는 player=-1 이 됨 
# player = np.random.choice([1,-1])
player = -1

def normalize_board(board, env, player):
    state_ = env.get_perspective_state(board,player)
    state = torch.from_numpy(state_).float().to(device)
    state = state.unsqueeze(0).unsqueeze(0)
    return state

def select_action(state, env, agent):
    print("output:\n")
    print("a prob:", agent.model(state)[0])
    print("value:", agent.model(state)[1])
    action_prob = agent.model(state)[0].detach().cpu().numpy()
    valid_actions = env.get_valid_actions(state[0][0].detach().cpu().numpy())
    action_prob *= valid_actions
    print(action_prob)
    action = np.argmax(action_prob)
    return action

def print_board_while_gaming(board, pointer, player):
    # os.system('cls' if os.name == 'nt' else 'clear')
    n_row, n_col = 6,7
    print("Connect Four")
    print("Player1 {} with C={},search={}: {}".format(model1_name,args['C'],args["num_searches"],X_mark))
    if do_I_play:
        print("Player2:",O_mark)
    else:
        print("Player2 {} with C={},search={}: {}".format(model2_name,args2['C'],args2["num_searches"],O_mark))
    print("-----------------------")
    empty_space = [" "]*n_col
    empty_space[pointer] = X_mark if player == 1 else O_mark
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



CF = ConnectFour()
folder_path = "model/alphazero/"
print("what the...")




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AlphaZeroResNet(nb1, hl1).to(device)
model.load_state_dict(torch.load(folder_path+model1_name, map_location=device))
model.eval()

mcts = MCTS(CF, args, model)

state = CF.get_initial_state()

pointer = 3
op_pointer = 3

if not do_I_play:
    model2 = AlphaZeroResNet(nb2,hl2).to(device)
    model2.load_state_dict(torch.load(folder_path+model2_name, map_location=device))
    model2.eval()

    
    mcts2 = MCTS(CF, args2, model2)
     
print_board_while_gaming(state,pointer,player)

while True:
    
    if player == -1:
        if not do_I_play:
            neutral_state = CF.change_perspective(state, player)
            mcts_probs = mcts2.search(neutral_state)
            print("player -1:",mcts_probs)
            # time.sleep(5)
            if greedy:
                action = np.argmax(mcts_probs)
            else:
                mcts_temp_probs = mcts_probs ** (1/temperature)
                mcts_temp_probs /= mcts_temp_probs.sum()
                action = np.random.choice(range(7),p=mcts_temp_probs)
        else:
            # print("state", state)
            # print("state", state[0][0].detach().cpu().numpy())
            valid_moves = CF.get_valid_moves(state)
            print("valid_moves", [i for i in range(CF.action_size) if valid_moves[i] == 1])
            action = int(input(f"{player}:"))

            if valid_moves[action] == 0:
                print("action not valid")
                continue

            
    elif player == 1:
        neutral_state = CF.change_perspective(state, player)
        mcts_probs = mcts.search(neutral_state)
        print("player 1:",mcts_probs)
        # time.sleep(5)
        if greedy:
            action = np.argmax(mcts_probs)
        else:
            mcts_temp_probs = mcts_probs ** (1/temperature)
            mcts_temp_probs /= mcts_temp_probs.sum()
            action = np.random.choice(range(7),p=mcts_temp_probs)
        
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

    print_board_while_gaming(state,pointer,player)

# CF = AlphaZeroenv.CFEnvforAlphaZero()

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# # 나중에 RandomAgent가 추가된다면 ConnectFourDQNAgent(), RandomAgent(), HeuristicAgent() 등등 선택 가능
# #agent = env.HeuristicAgent()
# config = get_model_config()


# pointer = 3
# new_pointer = 3

# op_pointer = 3

# player = np.random.choice([1,-1])
# print_board_while_gaming(state, pointer,player)

# is_terminal = False
# action = None
# while True:
#     if player==1:

#         if keyboard.is_pressed("down"):
#             if state[0,pointer] != 0:
#                 print("이미 가득 찬 곳을 선택하셨습니다. 다시 선택해주세요")
#                 time.sleep(0.1)
#                 continue
            
#             state = CF.get_next_state(state, pointer, player)
#             print_board_while_gaming(state,pointer,player)
#             player = CF.get_opponent(player)

#             time.sleep(0.1)
            
#         # move left
#         if keyboard.is_pressed("left"):
#             if pointer>0: pointer -= 1
#             print_board_while_gaming(state,pointer,player)

#             time.sleep(0.1)
#         # move right
#         if keyboard.is_pressed("right"):
#             if pointer<6: pointer += 1
#             print_board_while_gaming(state, pointer,player)

#             time.sleep(0.1)

    
    

#     else:
#         print_board_while_gaming(state, op_pointer,player)
#         thinking_time = random.normalvariate(1.2,0.4)
#         if thinking_time<0: continue

#         time.sleep(thinking_time)
#         neutral_state = CF.change_perspective(state, player)
#         mcts_probs = mcts.search(neutral_state)
#         action = np.argmax(mcts_probs)
#         print(action)
        
#         while op_pointer != action:
#             thinking_time = random.normalvariate(0.3,0.5)
#             if thinking_time<0 or thinking_time>0.6: continue
#             op_pointer += 1 if op_pointer < action else -1
#             print_board_while_gaming(state, op_pointer,player)
            
#             time.sleep(thinking_time)
#         thinking_time = random.normalvariate(0.5,0.15)
#         time.sleep(thinking_time)

#         board = CF.get_next_state(state, op_pointer, player)
#         op_pointer = action
#         print_board_while_gaming(state, op_pointer,player)
#         player = CF.get_opponent(player)

#     value, is_terminal = CF.get_value_and_terminated(state, action)
    
#     if is_terminal:
#         print(state)
#         if value == 1:
#             print(player, "won")
#         else:
#             print("draw")
#         break



        
        

    


 
