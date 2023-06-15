import numpy as np
import os
import random
import copy
import math
from collections import deque
import torch
from torch import nn, optim
import time
# models.py 분리 후 이동, 정상 작동하면 지울 듯 
# import torch.nn.init as init
# import torch.nn.functional as F
from models import *
from AlphaZeroenv import MCTS, CFEnvforAlphaZero


models = {
            1:CFLinear,
            2:CFCNN,
            3:HeuristicModel,
            4:RandomModel,
            5:AlphaZeroResNet,
            6:ResNetforDQN,
            7:CNNforMinimax,
        }

# console 창을 비우는 함수 
def clear():
    os.system('cls' if os.name=='nt' else 'clear')

# env의 board를 normalize 해주는 함수 
# 2를 -1로 바꿔서 board를 -1~1로 바꿔줌
def board_normalization(noise,env, model_type):
    # cnn을 사용하지 않는다면, 2차원 board를 1차원으로 바꿔줘야됨 
    if model_type == "Linear":
        arr = copy.deepcopy(env.board.flatten())
    elif model_type == "CNN": 
        arr = copy.deepcopy(env.board)


    """Replace all occurrences of 2 with -1 in a numpy array"""
    arr[arr == 2] = -1
    
    # 2p이면 보드판을 반전시켜서 보이게 하여, 항상 같은 색깔을 보면서 학습 가능
    if env.player == 2: arr = -1 * arr

    if noise:
        arr += np.random.randn(*arr.shape)/100.0

    return arr

# 두 모델의 승률을 비교하는 함수
# n_battle 만큼 서로의 policy로 대결하여 
# [model1's win, model2's win, draw] 리스트를 리턴 
def compare_model(model1, model2, n_battle=10):
    # epsilon을 복원하지 않으면, 학습 내내 고정됨 
    eps1 = model1.eps
    model1.eps = 0  # no exploration
    players = {1:model1, 2:model2}
    records = [0,0,0]  # model1 win, model2 win, draw
    comp_env = ConnectFourEnv()

    for round in range(n_battle):
        comp_env.reset()

        while not comp_env.done:
            # 성능 평가이므로, noise를 주지 않음 
            turn = comp_env.player
            state_ = board_normalization(noise=False,env=comp_env, model_type=players[turn].policy_net.model_type)
            state = torch.from_numpy(state_).float()
            
            action = players[turn].select_action(state, valid_actions=comp_env.valid_actions, player=turn)
            if isinstance(action, tuple):
                action = action[0]
            comp_env.step(action)
        
        if comp_env.win == 1: records[0] += 1
        elif comp_env.win == 2: records[1] += 1
        else: records[2] += 1

    model1.eps = eps1  # restore exploration

    return records

# model1과 model2의 policy에 따라
# 어떻게 플레이 하는지를 직접 확인가능
def simulate_model(model1, model2):
    eps1 = model1.eps
    model1.eps = 0  # no exploration
    players = {1:model1, 2:model2}

    test_env = ConnectFourEnv()

    test_env.reset()
    while not test_env.done:
        turn = test_env.player
        state_ = board_normalization(noise=False, env=test_env, model_type=players[turn].policy_net.model_type)
        state = torch.from_numpy(state_).float()

        action = players[turn].select_action(state, valid_actions=test_env.valid_actions, player=turn)
        test_env.step(action)
        test_env.print_board(clear_board=False)
        print("{}p put piece on {}".format(turn, action))
        time.sleep(1)   
    print("winner is {}p".format(test_env.win))

    model1.eps = eps1  # restore exploration

# 가장 기본적인 connect4 게임 환경 
class ConnectFourEnv:
    def __init__(self, n_row=6, n_col=7,first_player=None):
        self.n_row = n_row
        self.n_col = n_col
        self.board = np.zeros((n_row, n_col))
        if first_player is None:
            self.first_player = np.random.choice([1,2])
            self.player = self.first_player
        else: 
            self.first_player = first_player
            self.player = first_player

        # 만약 경기가 끝나면 win은 player 가 됨, 비길 경우 3 
        self.win = 0
        self.done = False
        # 가능한 actions들. 꽉찬 column엔 piece를 더 넣을 수 없기 때문 
        self.valid_actions = [i for i in range(self.n_col)]

    # board_normalization() 함수로 대체 예정 
    # def reverse_piece(self, board):
    #     board = np.where(
    #             board == 1,2, \
    #             np.where(board == 2,1, board)
    #     )
    #     return board

    # 게임이 끝났을 때 새로운 환경을 생성하는 대신 reset()으로 처리 
    def reset(self, first_player=None):
        self.board = np.zeros((self.n_row, self.n_col))
        if first_player is None: 
            self.first_player = np.random.choice([1,2])
            self.player = self.first_player
        else:
            self.first_player = first_player 
            self.player = first_player

        self.win = 0
        self.done = False
        self.valid_actions = [i for i in range(self.n_col)]

    # col에 조각 떨어뜨리기 
    def step(self, action):
        col = action
        # 경기가 끝나지 않을 때 negative reward 를 줄지 말지는 생각이 필요함 
        reward = 0.
        # reward = 0
        # 떨어뜨리려는 곳이 이미 가득 차있을 때
        # 로직을 바꿔서 이젠 이 if문은 실행되지 않을 것임 
        if not self.board[0,col] == 0:
            reward = -1
            # print(self.board, action)
            print("1:this cannot be happened")
        else:
            # piece를 둠 
            for row in range(self.n_row-1,-1,-1):
                if self.board[row][col] == 0:
                    self.board[row][col] = self.player
                    break
                else: continue
        
        # action을 취한 후 해당 column이 꽉차면 valid_action에서 제외 
        if self.board[0,col] != 0:
            if col in self.valid_actions:
                self.valid_actions.remove(col)
        
        # action을 취한 후 승패 체크
        self.check_win()
        if self.win != 0:
            # 비기면 0점 (비겼을 때의 reward도 생각해봐야됨)
            if self.win == 3: reward = 0
            # 이기면 +1점 
            elif self.player == self.win: reward = 1
            # 진 agent에겐 train 과정에서 따로 negative reward를 부여하므로
            # 해당 elif 문은 작동하지 않음 
            elif self.player != self.win: 
                reward = -1
                print("2:this cannot be happened")

        # 모든 행동이 완료되면 player change
        self.change_player()

        return (self.board, reward, self.done)

    def possible_next_states(self):
        possible_states = []
        for col in range(self.n_col):
            cpy_board = copy.deepcopy(self.board)
            if not cpy_board[0,col] == 0:
                possible_states.append(cpy_board)
            else:
                for row in range(self.n_row-1,-1,-1):
                    if cpy_board[row][col] == 0:
                        cpy_board[row][col] = self.player
                        break
                    else: continue
                possible_states.append(cpy_board)
        return possible_states

    # player를 change
    def change_player(self):
        self.player = int(2//self.player)

    # chatgpt에게 물어본 보드 출력 함수를 살짝 수정 
    # clear_board는 board를 출력하기 전에 console창을 비울지 여부 
    def print_board(self, clear_board=True):
        if clear_board: clear()
        board = copy.deepcopy(self.board)
        # if self.player == 2:
        #     board = self.reverse_piece(board)

        for row in range(self.n_row):
            row_str = "|"
            for col in range(self.n_col):
                if board[row][col] == 0:
                    row_str += " "
                elif board[row][col] == 1:
                    row_str += "X"
                elif board[row][col] == 2:
                    row_str += "O"
                row_str += "|"
            print(row_str)
        print("+" + "-" * (len(board[0]) * 2 - 1) + "+")
        print("player {}'s turn!".format(int(self.player)))

    # (x,y) 에서 8방향으로 적절한 좌표 3개를 제공 
    # def coor_for_8_direction(self, x, y):
    #     coors = []
    #     left, right, up, down = (False,)*4
    #     # (x,y) 기준 오른쪽
    #     if y+3<self.n_col:
    #         right=True
    #         coors.append(((x,y+1),(x,y+2),(x,y+3)))
    #     # (x,y) 기준 왼쪽
    #     if y-3>=0:
    #         left=True
    #         coors.append(((x,y-1),(x,y-2),(x,y-3)))
    #     # (x,y) 기준 위 
    #     if x-3>=0:
    #         up=True
    #         coors.append(((x-1,y),(x-2,y),(x-3,y)))
    #     # (x,y) 기준 아래 
    #     if x+3<self.n_row:
    #         down=True
    #         coors.append(((x+1,y),(x+2,y),(x+3,y)))
    #     # (x,y) 기준 오른쪽 위 
    #     if right and up:
    #         coors.append(((x-1,y+1),(x-2,y+2),(x-3,y+3)))
    #     # (x,y) 기준 오른쪽 아래 
    #     if right and down:
    #         coors.append(((x+1,y+1),(x+2,y+2),(x+3,y+3)))
    #     # (x,y) 기준 왼쪽 위 
    #     if left and up:
    #         coors.append(((x-1,y-1),(x-2,y-2),(x-3,y-3)))
    #     # (x,y) 기준 왼쪽 아래 
    #     if left and down:
    #         coors.append(((x+1,y-1),(x+2,y-2),(x+3,y-3)))

    #     return coors 


    # 승패가 결정됐는지 확인하는 함수
    # 0: not end
    # 1: player 1 win
    # 2: player 2 win
    # 3: draw 
    # def check_win(self):

    #     for x in range(self.n_row-1,-1,-1):
    #         for y in range(self.n_col):
    #             piece = self.board[x][y]
    #             if piece == 0: continue

    #             coor_list = self.coor_for_8_direction(x,y)
    #             for coors in coor_list:
    #                 # print("coors:",coors)
    #                 if piece == self.board[coors[0]] == self.board[coors[1]] == self.board[coors[2]]:
    #                     self.win = piece
    #                     self.done = True
    #                     return

    #     if not 0 in self.board[0,:]:
    #         self.win = 3
        
    #     if self.win != 0: self.done = True

    # made by chatgpt and I edit little bit.
    # 가로, 세로, 대각선에 완성된 줄이 있는지를 체크한다 
    def check_win(self):
        for i in range(self.n_row):
            for j in range(self.n_col):
                if self.board[i][j] == self.player:
                    # horizontal
                    if j + 3 < self.n_col and self.board[i][j+1] == self.board[i][j+2] == self.board[i][j+3] == self.player:
                        self.win = self.player
                        self.done = True
                        return
                    # vertical
                    if i + 3 < self.n_row and self.board[i+1][j] == self.board[i+2][j] == self.board[i+3][j] == self.player:
                        self.win = self.player
                        self.done = True
                        return
                    # diagonal (down right)
                    if i + 3 < self.n_row and j + 3 < self.n_col and self.board[i+1][j+1] == self.board[i+2][j+2] == self.board[i+3][j+3] == self.player:
                        self.win = self.player
                        self.done = True
                        return
                    # diagonal (up right)
                    if i - 3 >= 0 and j + 3 < self.n_col and self.board[i-1][j+1] == self.board[i-2][j+2] == self.board[i-3][j+3] == self.player:
                        self.win = self.player
                        self.done = True
                        return
    
        # no winner
        # 맨 윗줄이 모두 꽉차있다면, 비긴 것
        if not 0 in self.board[0,:]:
            self.win = 3  # 3 means the game is a draw
            self.done = True

        


    def step_human(self, col):
        self.step(col)
        self.print_board()

    # 그냥 random으로 두고 싶을 때 
    def step_cpu(self):
        self.step(np.random.choice(range(self.n_col)))
        self.print_board()





    

