import env
import AlphaZeroenv
import numpy as np
import torch
import time
import os
import copy
import random
import keyboard
from agent_structure import ConnectFourDQNAgent, MinimaxAgent, AlphaZero
from functions import get_model_config, get_model_and_config_name, get_encoded_state



import lovely_tensors as lt
lt.monkey_patch()

X_mark = "\033[31mX\033[0m"
O_mark = "\033[33mO\033[0m"

def print_board_while_gaming(env, pointer, board=None):
    
    print("Connect Four")
    print("Player1: "+X_mark)
    print("Player2: "+O_mark)
    print("-----------------------")
    empty_space = [" "]*env.n_col
    empty_space[pointer] = X_mark if env.player == 1 else O_mark
    if board is None:
        board = copy.deepcopy(env.board)
    else:
        empty_space[pointer] = " "
    
        # if self.player == 2:
        #     board = self.reverse_piece(board)
    
    
    
    row_str = " "
    for col in range(env.n_col):
        row_str += empty_space[col]
        row_str += " "
    print(row_str)
    for row in range(env.n_row):
        row_str = "|"
        for col in range(env.n_col):
            if board[row][col] == 0:
                row_str += " "
            elif board[row][col] == 1:
                row_str += X_mark
            elif board[row][col] == 2:
                row_str += O_mark
            row_str += "|"
        print(row_str)
    print("+" + "-" * (len(board[0]) * 2 - 1) + "+")
    print("player {}'s turn!".format(int(env.player)))





CF = env.ConnectFourEnv() 
mode = input("to play with human, type 'human'(else just enter):")

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 나중에 RandomAgent가 추가된다면 ConnectFourDQNAgent(), RandomAgent(), HeuristicAgent() 등등 선택 가능
#agent = env.HeuristicAgent()
config = get_model_config()


model_name, config_name = get_model_and_config_name('model/model_for_play')
prev_model_config = get_model_config('model/model_for_play/'+config_name)
kwargs={
    'use_conv':prev_model_config['use_conv'], \
    'use_minimax':prev_model_config['use_minimax'], \
    'use_resnet':prev_model_config['use_resnet'],
    'next_state_is_op_state':prev_model_config['next_state_is_op_state']
}
agent = ConnectFourDQNAgent(**kwargs)
agent.eps = 0

agent.policy_net.load_state_dict(torch.load('model/model_for_play/'+model_name, map_location=device))
agent.update_target_net() 
  

# agent = MinimaxAgent()
pointer = 3
new_pointer = 3

op_pointer = 3
print_board_while_gaming(CF, pointer)
while CF.win==0:
    if CF.player==1:

        if keyboard.is_pressed("down"):
            if CF.board[0,pointer] != 0:
                print("이미 가득 찬 곳을 선택하셨습니다. 다시 선택해주세요")
                time.sleep(0.1)
                continue
            
            CF.step_human(pointer)
            print_board_while_gaming(CF, pointer)
            time.sleep(0.1)
            
        # move left
        if keyboard.is_pressed("left"):
            if pointer>0: pointer -= 1
            print_board_while_gaming(CF, pointer)

            time.sleep(0.1)
        # move right
        if keyboard.is_pressed("right"):
            if pointer<CF.n_col-1: pointer += 1
            print_board_while_gaming(CF, pointer)

            time.sleep(0.1)

    
    

    else:
        print_board_while_gaming(CF, op_pointer)
        thinking_time = random.normalvariate(1.2,0.4)
        if thinking_time<0: continue

        time.sleep(thinking_time)
        state_ = env.board_normalization(False,CF, agent.policy_net.model_type)
        if agent.use_conv:
            state = torch.tensor(get_encoded_state(state_))
        else: state = torch.from_numpy(state_).float()
        action = agent.select_action(state, CF, player=CF.player)
        # print(agent.policy_net(state))
        print(action)
        if isinstance(action, tuple): action = action[0]
        
        while op_pointer != action:
            thinking_time = random.normalvariate(0.3,0.5)
            if thinking_time<0 or thinking_time>0.6: continue
            op_pointer += 1 if op_pointer < action else -1
            print_board_while_gaming(CF, op_pointer)
            
            time.sleep(thinking_time)
        thinking_time = random.normalvariate(0.5,0.15)
        time.sleep(thinking_time)
        
        CF.step(action)
        op_pointer = action
        print_board_while_gaming(CF, pointer)
        

    

    

print_board_while_gaming(CF, pointer)
            
if CF.win==3:
    print("draw!")

else: print("player {} win!".format(int(CF.win)))
    

                
# while CF.win==0:
#     if CF.player==1:
#         col = int(input("어디에 둘 지 고르세요[0~{}]:".format(CF.n_col-1)))
#         if col>=CF.n_col or col<0:
#             print("잘못된 숫자입니다. 다시 골라주세요")
#             continue
#         elif CF.board[0,col] != 0:
#             print("이미 가득 찬 곳을 선택하셨습니다. 다시 선택해주세요")
#             continue

#         CF.step_human(col)
#     else:
#         # random agent가 추가되면 step_cpu() 를 사용하지 않아도 될듯
#         # 현재 cpu는 random
#         #CF.step_cpu()
#         time.sleep(1)
#         state_ = env.board_normalization(False,CF, agent.policy_net.model_type)
#         state = torch.from_numpy(state_).float()
#         action = agent.select_action(state, valid_actions=CF.valid_actions, player=CF.player)
#         if isinstance(action, tuple): action = action[0]
#         CF.step(action)
#         CF.print_board()


        
# if CF.win==3:
#     print("draw!")

# else: print("player {} win!".format(int(CF.win)))

