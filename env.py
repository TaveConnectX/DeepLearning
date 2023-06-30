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
from functions import get_valid_actions, get_next_board

# models = {
#             1:CFLinear,
#             2:CFCNN,
#             3:HeuristicModel,
#             4:RandomModel,
#             5:AlphaZeroResNet,
#             6:ResNetforDQN,
#             7:CNNforMinimax,
#         }

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
    temp = model1.temp
    model1.eps = 0  # no exploration
    model1.temp = 0
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
            
            action = players[turn].select_action(state, comp_env, player=turn)
            if isinstance(action, tuple):
                action = action[0]
            comp_env.step(action)
        
        if comp_env.win == 1: records[0] += 1
        elif comp_env.win == 2: records[1] += 1
        else: records[2] += 1

    model1.eps = eps1  # restore exploration
    model1.temp = temp
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

        action = players[turn].select_action(state, test_env, player=turn)
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
            exit()
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
        # self.print_board()

    # 그냥 random으로 두고 싶을 때 
    def step_cpu(self):
        self.step(np.random.choice(range(self.n_col)))
        self.print_board()

# ConnectFourEnv와는 다르게 게임 규칙만 들어가 있음 
# player는 1과 -1로 이루어져 있으므로 정규화할 필요 없음 
# self play 이므로 초기 player를 랜덤으로 둘 필요 없음 
# action은 one-hot encoding 된 상태로 받음 ex. [0,0,1,0,0,0,0]

class ConnectFour:
    def __init__(self):
        self.row_count = 6
        self.column_count = 7
        self.action_size = self.column_count
        self.in_a_row = 4
        
    def __repr__(self):
        return "ConnectFour"
        
    def get_initial_state(self):
        return np.zeros((self.row_count, self.column_count))
    
    def get_next_state(self, state, action, player):
        row = np.max(np.where(state[:, action] == 0))
        state[row, action] = player
        return state
    
    def get_valid_moves(self, state):
        return (state[0] == 0).astype(np.uint8)
    
    def check_win(self, state, action):
        if action == None:
            return False
        
        row = np.min(np.where(state[:, action] != 0))
        column = action
        player = state[row][column]

        def count(offset_row, offset_column):
            for i in range(1, self.in_a_row):
                r = row + offset_row * i
                c = action + offset_column * i
                if (
                    r < 0 
                    or r >= self.row_count
                    or c < 0 
                    or c >= self.column_count
                    or state[r][c] != player
                ):
                    return i - 1
            return self.in_a_row - 1

        return (
            count(1, 0) >= self.in_a_row - 1 # vertical
            or (count(0, 1) + count(0, -1)) >= self.in_a_row - 1 # horizontal
            or (count(1, 1) + count(-1, -1)) >= self.in_a_row - 1 # top left diagonal
            or (count(1, -1) + count(-1, 1)) >= self.in_a_row - 1 # top right diagonal
        )
    
    def get_value_and_terminated(self, state, action):
        if self.check_win(state, action):
            return 1, True
        if np.sum(self.get_valid_moves(state)) == 0:
            return 0, True
        return 0, False
    
    def get_opponent(self, player):
        return -player
    
    def get_opponent_value(self, value):
        return -value
    
    def change_perspective(self, state, player):
        return state * player
    
    def get_encoded_state(self, state):
        encoded_state = np.stack(
            (state == -1, state == 0, state == 1)
        ).astype(np.float32)
        
        if len(state.shape) == 3:
            encoded_state = np.swapaxes(encoded_state, 0, 1)
        
        return encoded_state
    

class Node:
    def __init__(self, game, args, state, parent=None, action_taken=None, prior=0, visit_count=0):
        self.game = game
        self.args = args
        self.state = state
        self.parent = parent
        self.action_taken = action_taken
        self.prior = prior
        
        self.children = []
        
        self.visit_count = visit_count
        self.value_sum = 0
        
    def is_fully_expanded(self):
        return len(self.children) > 0
    
    def select(self):
        best_child = None
        best_ucb = -np.inf
        
        for child in self.children:
            ucb = self.get_ucb(child)
            if ucb > best_ucb:
                best_child = child
                best_ucb = ucb
                
        return best_child
    
    def get_ucb(self, child):
        if child.visit_count == 0:
            q_value = 0
        else:
            q_value = 1 - ((child.value_sum / child.visit_count) + 1) / 2
        return q_value + self.args['C'] * (math.sqrt(self.visit_count) / (child.visit_count + 1)) * child.prior
    
    def expand(self, policy):
        for action, prob in enumerate(policy):
            if prob > 0:
                child_state = self.state.copy()
                child_state = self.game.get_next_state(child_state, action, 1)
                child_state = self.game.change_perspective(child_state, player=-1)

                child = Node(self.game, self.args, child_state, self, action, prob)
                self.children.append(child)
                
        return child
            
    def backpropagate(self, value):
        self.value_sum += value
        self.visit_count += 1
        
        value = self.game.get_opponent_value(value)
        if self.parent is not None:
            self.parent.backpropagate(value)  


class MCTS:
    def __init__(self, game, args, model):
        self.game = game
        self.args = args
        self.model = model
        
    @torch.no_grad()
    def search(self, state):
        root = Node(self.game, self.args, state, visit_count=1)
        
        policy, _ = self.model(
            torch.tensor(self.game.get_encoded_state(state), device=self.model.device).unsqueeze(0)
        )
        policy = torch.softmax(policy, axis=1).squeeze(0).cpu().numpy()
        policy = (1 - self.args['dirichlet_epsilon']) * policy + self.args['dirichlet_epsilon'] \
            * np.random.dirichlet([self.args['dirichlet_alpha']] * self.game.action_size)
        
        valid_moves = self.game.get_valid_moves(state)
        policy *= valid_moves
        policy /= np.sum(policy)
        root.expand(policy)
        
        for search in range(self.args['num_searches']):
            node = root
            
            while node.is_fully_expanded():
                node = node.select()
                
            value, is_terminal = self.game.get_value_and_terminated(node.state, node.action_taken)
            value = self.game.get_opponent_value(value)
            
            if not is_terminal:
                policy, value = self.model(
                    torch.tensor(self.game.get_encoded_state(node.state), device=self.model.device).unsqueeze(0)
                )
                policy = torch.softmax(policy, axis=1).squeeze(0).cpu().numpy()
                valid_moves = self.game.get_valid_moves(node.state)
                policy *= valid_moves
                policy /= np.sum(policy)
                
                value = value.item()
                
                node.expand(policy)
                
            node.backpropagate(value)    
            
            
        action_probs = np.zeros(self.game.action_size)
        for child in root.children:
            action_probs[child.action_taken] = child.visit_count
        action_probs /= np.sum(action_probs)
        return action_probs
    
class MCTSParallel:
    def __init__(self, game, args, model):
        self.game = game
        self.args = args
        self.model = model
        
    @torch.no_grad()
    def search(self, states, spGames):
        # print(self.game.get_encoded_state(states).shape)
        policy, _ = self.model(
            torch.tensor(self.game.get_encoded_state(states), device=self.model.device)
        )
        policy = torch.softmax(policy, axis=1).cpu().numpy()
        policy = (1 - self.args['dirichlet_epsilon']) * policy + self.args['dirichlet_epsilon'] \
            * np.random.dirichlet([self.args['dirichlet_alpha']] * self.game.action_size, size=policy.shape[0])
        
        for i, spg in enumerate(spGames):
            spg_policy = policy[i]
            valid_moves = self.game.get_valid_moves(states[i])
            spg_policy *= valid_moves
            spg_policy /= np.sum(spg_policy)

            spg.root = Node(self.game, self.args, states[i], visit_count=1)
            spg.root.expand(spg_policy)
        
        for search in range(self.args['num_searches']):
            for spg in spGames:
                spg.node = None
                node = spg.root

                while node.is_fully_expanded():
                    node = node.select()

                value, is_terminal = self.game.get_value_and_terminated(node.state, node.action_taken)
                value = self.game.get_opponent_value(value)
                
                if is_terminal:
                    node.backpropagate(value)
                    
                else:
                    spg.node = node
                    
            expandable_spGames = [mappingIdx for mappingIdx in range(len(spGames)) if spGames[mappingIdx].node is not None]
                    
            if len(expandable_spGames) > 0:
                states = np.stack([spGames[mappingIdx].node.state for mappingIdx in expandable_spGames])
                
                policy, value = self.model(
                    torch.tensor(self.game.get_encoded_state(states), device=self.model.device)
                )
                policy = torch.softmax(policy, axis=1).cpu().numpy()
                value = value.cpu().numpy()
                
            for i, mappingIdx in enumerate(expandable_spGames):
                node = spGames[mappingIdx].node
                spg_policy, spg_value = policy[i], value[i]
                
                valid_moves = self.game.get_valid_moves(node.state)
                spg_policy *= valid_moves
                spg_policy /= np.sum(spg_policy)

                node.expand(spg_policy)
                node.backpropagate(spg_value)


class SPG:
    def __init__(self, game):
        self.state = game.get_initial_state()
        self.memory = []
        self.root = None
        self.node = None

# class Node():
#     def __init__(self, data=None, children={}, player=None):
#         self.data = data
#         self.parent = None
#         self.children = children
#         self.player = None


# class Tree():
#     def __init__(self, state=None):
#         self.root = Node(data=state)
#         self.cur = self.root

#     def branch(self, node):
#         valid_actions = get_valid_actions(node.data)
#         for action in valid_actions:
#             next_board = get_next_board(node.data, action)
#             node.children[action] = Node(next_board, player=2//node.player)
#             node.children[action].parent = node

#     def set_root(self, node):
#         node.parent = None
#         self.root = node
#         self.cur = self.root

#     def get_root(self):
#         self.cur = self.root

#     def get_parent(self):
#         self.cur = self.cur.parent


