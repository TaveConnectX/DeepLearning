import numpy as np
import os
import random
import copy
from collections import deque
import torch
from torch import nn, optim
import time
# models.py 분리 후 이동, 정상 작동하면 지울 듯 
# import torch.nn.init as init
# import torch.nn.functional as F
from models import *


models = {
            1:CFLinear,
            2:CFCNN,
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
        reward = -1/43.
        # reward = 0
        # 떨어뜨리려는 곳이 이미 가득 차있을 때
        # 로직을 바꿔서 이젠 이 if문은 실행되지 않을 것임 
        if not self.board[0,col] == 0:
            reward = -1
            print("this cannot be happened")
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
                print("this cannot be happened")

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

# this class is for Q-learning, but I don't use it
class CFAgent:
    def __init__(self, env, lr=0.1, gamma=0.9, epsilon=0.1, policy='q-learning'):
        self.env = env
        self.n_states = np.prod(env.board.shape)
        self.n_actions = env.n_col
        self.q_table = np.zeros((self.n_states, self.n_actions))
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.policy = policy
    
    def get_action(self, state):
        if self.policy == 'random':
            action = random.randint(0, self.n_actions-1)

        elif self.policy == 'q-learning':
            # 엡실론-그리디 정책에 따라 행동을 선택
            if random.uniform(0, 1) < self.epsilon:
                action = random.randint(0, self.n_actions - 1)
            else:
                action = np.argmax(self.q_table[state])

        return action
    
    
    def update(self, state, action, reward, next_state):
        # Q 테이블을 업데이트
        td_error = reward + self.gamma * np.max(self.q_table[next_state]) - self.q_table[state][action]
        self.q_table[state][action] += self.lr * td_error


# editable hyperparameters
# let iter=10000, then
# learning rate 0.01~0.0001
# batch_size 2^5~2^10
# hidden layer  2^6~2^9
# memory_len  100~10000
# target_update 1~1000

# op_update 0~2000 ( if 0, then opponent always act randomly)
class ConnectFourDQNAgent:
    def __init__(self, state_size=6*7, action_size=7, gamma=0.99, lr=0.0001, batch_size=256,hidden_size=64, target_update=100, eps=1., memory_len=2000):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.policy_net = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size)
        ).to(self.device)

        # He로 가중치 초기화
        for layer in self.policy_net:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
        
        self.target_net = copy.deepcopy(self.policy_net)
        
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = deque(maxlen=memory_len)
        self.gamma = gamma
        self.state_size = state_size
        self.action_size = action_size
        self.target_update = target_update
        self.steps = 0
        self.eps = eps
        self.batch_size = batch_size
        self.losses = []
        
    def select_action(self, state, valid_actions=None, player=1):
        if valid_actions is None:
            valid_actions = range(self.action_size)

        if np.random.uniform() < self.eps:
            
            return np.random.choice(valid_actions)
        with torch.no_grad():
            state_ = torch.FloatTensor(state).to(self.device)
            q_value = self.policy_net(state_)
            # print("state:",state)
            # print("valid_actions:",valid_actions)
            # print("q_value:",q_value)
            valid_q_values = q_value.squeeze()[torch.tensor(valid_actions)]
            return valid_actions[torch.argmax(valid_q_values)]
        
    def append_memory(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        minibatch = random.sample(self.memory, self.batch_size)

        
        state1_batch = torch.cat([s1 for (s1,a,r,s2,d) in minibatch]).to(self.device)
        action_batch = torch.Tensor([a for (s1,a,r,s2,d) in minibatch]).to(self.device)
        reward_batch = torch.Tensor([r for (s1,a,r,s2,d) in minibatch]).to(self.device)
        state2_batch = torch.cat([s2 for (s1,a,r,s2,d) in minibatch]).to(self.device)
        done_batch = torch.Tensor([d for (s1,a,r,s2,d) in minibatch]).to(self.device)
        Q1 = self.policy_net(state1_batch) 
        with torch.no_grad():
            Q2 = self.target_net(state2_batch) #B
        
        Y = reward_batch + self.gamma * ((1-done_batch) * torch.max(Q2,dim=1)[0])
        X = Q1.gather(dim=1,index=action_batch.long().unsqueeze(dim=1)).squeeze()
        loss = nn.MSELoss()(X, Y.detach())
        self.losses.append(loss.detach().cpu().numpy())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # state, action, reward, next_state, done = zip(*minibatch)
        # print(state)
        # state = torch.FloatTensor(state) # .to(self.device)

        # action = torch.LongTensor(action).unsqueeze(1)  #  .to(self.device)
        # reward = torch.FloatTensor(reward).unsqueeze(1)  # .to(self.device)
        # next_state = torch.FloatTensor(next_state)  # .to(self.device)
        # done = torch.FloatTensor(done).unsqueeze(1)  # .to(self.device)
        
        # current_q_values = self.policy_net(state).gather(1, action)
        # next_q_values = self.target_net(next_state).detach().max(1)[0].unsqueeze(1)
        # expected_q_values = reward + self.gamma * next_q_values * (1 - done)
        
        # loss = nn.MSELoss()(current_q_values, expected_q_values)
        # self.optimizer.zero_grad()
        # loss.backward()
        # self.optimizer.step()
        self.steps += 1
        if self.steps % self.target_update == 0:
            self.update_target_net()
        
    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())


    

class ConnectFourDQNAgent(nn.Module):
    def __init__(self, state_size=6*7, action_size=7, gamma=0.99, lr=0.001, batch_size=1024, target_update=2000, eps=1., memory_len=10000,model_num=1):
        super(ConnectFourDQNAgent,self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.policy_net = models[model_num]()
        # 실제 업데이트되는 network
        # self.policy_net = CFCNN()
        # target network
        self.target_net = copy.deepcopy(self.policy_net)
        # deepcopy하면 파라미터 load를 안해도 되는거 아닌가? 일단 두자
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()


        # optimizer는 기본적으로 adam을 사용하겠지만 추후 다른 것으로 실험할 수도 있음
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        
        # DQN에 사용될 replay memory(buffer)
        self.memory = deque(maxlen=memory_len)
        self.gamma = gamma  # discount factor
        self.state_size = state_size
        self.action_size = action_size
        self.target_update = target_update  # target net update 주기(여기선 step)
        self.steps = 0
        self.eps = eps  # DQN에서 사용될 epsilon
        self.batch_size = batch_size  # size of mini-batch
        self.losses = []  # loss들을 담는 list 

    # def forward(self, x):
    #     x = torch.tensor(x).float().to(self.device)
    #     x = self.conv_net(x)
    #     x = x.view(x.size(0), -1)
    #     x = self.fc(x)

    #     return x

    def select_action(self, state, valid_actions=None, player=1):
        if valid_actions is None:
            valid_actions = range(self.action_size)

        if np.random.uniform() < self.eps:
            return np.random.choice(valid_actions)
        with torch.no_grad():
            state_ = torch.FloatTensor(state).to(self.device)
            # CNN일 때만 차원을 바꿔줌 
            if self.policy_net.model_type == 'CNN':
                state_ = state_.unsqueeze(0).unsqueeze(0)  # (6,7) -> (1,1,6,7)
            q_value = self.policy_net(state_)
            # print("state:",state)
            # print("valid_actions:",valid_actions)
            # print("q_value:",q_value)
            valid_q_values = q_value.squeeze()[torch.tensor(valid_actions)]
            return valid_actions[torch.argmax(valid_q_values)]
        
    # replay buffer에 경험 추가 
    def append_memory(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def train(self,epi,env,op_model):
        env.reset()

        # models 딕셔너리는 전역 변수로 사용하므로, players로 변경 
        players = {1: self, 2: op_model}

        for i in range(epi):
            # 100번마다 loss, eps 등의 정보 표시
            if i!=0 and i%100==0: 
                env.print_board(clear_board=False)
                print("epi:",i, ", agent's step:",self.steps)
                # 얼마나 학습이 진행되었는지 확인하기 위해, 모델 성능 측정 
                record = compare_model(self, op_model, n_battle=100)
                print(record)
                print("agent의 승률이 {}%".format(int(100*record[0]/sum(record))))
                print("loss:",sum(self.losses[-101:-1])/100.)
                print("epsilon:",self.eps)
                # simulate_model() 을 실행시키면 직접 행동 관찰 가능
                # simulate_model(self, op_model)

            
            env.reset()

            state_ = board_normalization(noise=True, env=env, model_type=players[env.player].policy_net.model_type)
            state = torch.from_numpy(state_).float()
            done = False

            past_state, past_action, past_reward, past_done = state, None, None, done
            
            while not done:
                # 원래는 player, op_player 였지만, 직관적인 이해를 위해 수정 
                turn = env.player
                op_turn = 2//turn

                action = players[turn].select_action(state, valid_actions=env.valid_actions, player=turn)

                observation, reward, done = env.step(action)
                op_state_ = board_normalization(noise=True, env=env, model_type=players[turn].policy_net.model_type)
                op_state = torch.from_numpy(op_state_).float() 

                if past_action is not None:  # 맨 처음이 아닐 때 
                    # 경기가 끝났을 때(중요한 경험)
                    if done:
                        repeat = 1
                        # 중요한 경험일 때는 더 많이 memory에 추가해준다(optional)
                        if reward > 0: repeat = 5
                        for j in range(repeat):
                            # 돌을 놓자마자 끝났으므로, next_state가 반전됨, 따라서 -1을 곱해준다
                            if turn==1:
                                self.append_memory(state,action, reward, op_state*-1, done)
                                # print for debugging
                                # print("for player")
                                # print("state:\n",torch.round(state).reshape(6,7).int())
                                # print("action:",action)
                                # print("reward:",reward)
                                # print("next_state\n",torch.round(op_state*-1).reshape(6,7).int())
                                # print()
                            # 내가 이겼으므로 상대는 음의 보상을 받음 
                            #Qmodels[op_player].append_memory(past_state, past_action, -reward, op_state, done)
                            if turn==2:
                                self.append_memory(past_state, past_action, -reward, op_state, done)
                                # print for debugging
                                # print("for opponent")
                                # print("state:\n",torch.round(past_state).reshape(6,7).int())
                                # print("action:",past_action)
                                # print("reward:",-reward)
                                # print("next_state\n",torch.round(op_state).reshape(6,7).int())
                                # print()


                    # 경기가 끝나지 않았다면
                    elif turn==2:  # 내 경험만 수집한다
                        self.append_memory(past_state, past_action, past_reward, op_state, past_done)
                        # print for debugging
                        print("for opponent")
                        print("state:\n",torch.round(past_state).reshape(6,7).int())
                        print("action:",past_action)
                        print("reward:",past_reward)
                        print("next_state\n",torch.round(op_state).reshape(6,7).int())
                        print()

                
                # op_action = Qmodels[player].select_action(op_state,valid_actions=CFenv.valid_actions, player=player)
                # op_observation, op_reward, op_done = CFenv.step(op_action)
                
                # next_state_ = board_normalization(op_observation.flatten()) + np.random.randn(1, Qagent.state_size)/100.0
                # next_state = torch.from_numpy(next_state_).float()
                # # 2p가 돌을 놓자마자 끝남 
                # if op_done:
                #     Qmodels[player].append_memory(op_state,op_action, op_reward, next_state, op_done)
                #     Qmodels[op_player].append_memory(state,action, -op_reward, next_state, op_done)
                # else:
                #     exp = (state, action, reward, next_state, done)
                #     Qmodels[player].append_memory(*exp)

                # info들 업데이트 해줌 
                past_state = state
                past_action = action
                past_reward = reward
                past_done = done
                state = op_state
                
                # replay buffer 를 이용하여 mini-batch 학습
                self.replay()
                # if Qagent.memory and abs(Qagent.memory[-1][2])!=1:
                #     print("state:\n",torch.round(Qagent.memory[-1][0]).int())
                #     print("action:",Qagent.memory[-1][1])
                #     print("reward:",Qagent.memory[-1][2])
                #     print("next_state\n",torch.round(Qagent.memory[-1][3]).int())
                
                # 게임이 끝났다면 나가기 
                if done: break
            # print("eps:",Qagent.eps)
            # epsilon-greedy
            # min epsilon을 가지기 전까지 episode마다 조금씩 낮춰준다(1 -> 0.1)
            if self.eps > 0.1: self.eps -= (1/epi)

    # mini-batch로 업데이트 
    def replay(self):
        if len(self.memory) < self.batch_size*2:
            return
        # batch size 만큼 랜덤으로 꺼낸다 
        minibatch = random.sample(self.memory, self.batch_size)


        if self.policy_net.model_type == 'Linear':
            # state_batch.shape: (batch_size, 42)
            state1_batch = torch.stack([s1 for (s1,a,r,s2,d) in minibatch]).to(self.device)
            state2_batch = torch.stack([s2 for (s1,a,r,s2,d) in minibatch]).to(self.device)
        elif self.policy_net.model_type == 'CNN':
            # state_batch.shape: (batch_size, 1, 6, 7)
            state1_batch = torch.stack([s1 for (s1,a,r,s2,d) in minibatch]).unsqueeze(1).to(self.device)
            state2_batch = torch.stack([s2 for (s1,a,r,s2,d) in minibatch]).unsqueeze(1).to(self.device)

        # action_batch.shape: (batch_size, )
        action_batch = torch.Tensor([a for (s1,a,r,s2,d) in minibatch]).to(self.device)
        reward_batch = torch.Tensor([r for (s1,a,r,s2,d) in minibatch]).to(self.device)
        done_batch = torch.Tensor([d for (s1,a,r,s2,d) in minibatch]).to(self.device)
        
        # print("state1_batch:",state1_batch.shape)
        Q1 = self.policy_net(state1_batch)  # (256,7)
        with torch.no_grad():
            Q2 = self.target_net(state2_batch)
        # target Q value들 
        Y = reward_batch + self.gamma * ((1-done_batch) * torch.max(Q2,dim=1)[0])
        # 해당하는 action을 취한 q value들
        X = Q1.gather(dim=1,index=action_batch.long().unsqueeze(dim=1)).squeeze()
        
        
        loss = nn.MSELoss()(X, Y.detach())
        # print("to compare overestimation of Q value")
        # print(state1_batch[200][0])
        # print(state2_batch[200][0])
        # print("action:",action_batch[200])
        # print("reward:",reward_batch[200])
        # print(Q1[200])
        # print(Q2[200])
        # print()

        # tensor.numpy()는 cpu에서만 가능하므로 cpu() 처리
        self.losses.append(loss.detach().cpu().numpy())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # state, action, reward, next_state, done = zip(*minibatch)
        # print(state)
        # state = torch.FloatTensor(state) # .to(self.device)

        # action = torch.LongTensor(action).unsqueeze(1)  #  .to(self.device)
        # reward = torch.FloatTensor(reward).unsqueeze(1)  # .to(self.device)
        # next_state = torch.FloatTensor(next_state)  # .to(self.device)
        # done = torch.FloatTensor(done).unsqueeze(1)  # .to(self.device)
        
        # current_q_values = self.policy_net(state).gather(1, action)
        # next_q_values = self.target_net(next_state).detach().max(1)[0].unsqueeze(1)
        # expected_q_values = reward + self.gamma * next_q_values * (1 - done)
        
        # loss = nn.MSELoss()(current_q_values, expected_q_values)
        # self.optimizer.zero_grad()
        # loss.backward()
        # self.optimizer.step()
        self.steps += 1
        if self.steps % self.target_update == 0:
            print("update target net")
            self.update_target_net()

        
    # target net에 policy net 파라미터 들을 업데이트 해줌 
    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())







    



