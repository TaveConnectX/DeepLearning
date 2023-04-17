import numpy as np
import os
import random
import copy
from collections import deque
import torch
from torch import nn, optim

def clear():
    os.system('cls' if os.name=='nt' else 'clear')

class ConnectFourEnv:
    def __init__(self, n_row=6, n_col=7, mode='cpu'):
        self.n_row = n_row
        self.n_col = n_col
        self.board = np.zeros((n_row, n_col))
        self.mode = mode
        self.player = np.random.choice([1,2])
        self.win = 0
        self.done = False
        self.valid_actions = [i for i in range(self.n_col)]

    def reverse_piece(self, board):
        board = np.where(
                board == 1,2, \
                np.where(board == 2,1, board)
        )
        return board

    def reset(self):
        self.board = np.zeros((self.n_row, self.n_col))
        self.player = np.random.choice([1,2])
        self.win = 0
        self.done = False
        self.valid_actions = [i for i in range(self.n_col)]

    # col에 조각 떨어뜨리기 
    def step(self, action):
        col = action
        reward = 0
        # 떨어뜨리려는 곳이 이미 가득 차있을 때
        # 로직을 바꿔서 이젠 이 if문은 실행되지 않을 것임 
        if not self.board[0,col] == 0:
            reward = -1
            print("this cannot be happened")
        else:
            for row in range(self.n_row-1,-1,-1):
                if self.board[row][col] == 0:
                    self.board[row][col] = self.player
                    break
                else: continue
        
        if self.board[0,col] != 0:
            if col in self.valid_actions:
                self.valid_actions.remove(col)

        self.check_win()
        if self.win != 0:
            if self.win == 3: reward = -0.1
            elif self.player == self.win: reward = 1
            elif self.player != self.win: 
                reward = -1
                print("this cannot be happened")
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

    def change_player(self):
        self.player = int(2//self.player)

    # chatgpt에게 물어본 보드 출력 함수를 살짝 수정 
    def print_board(self):
        clear()
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
    def coor_for_8_direction(self, x, y):
        coors = []
        left, right, up, down = (False,)*4
        # (x,y) 기준 오른쪽
        if y+3<self.n_col:
            right=True
            coors.append(((x,y+1),(x,y+2),(x,y+3)))
        # (x,y) 기준 왼쪽
        if y-3>=0:
            left=True
            coors.append(((x,y-1),(x,y-2),(x,y-3)))
        # (x,y) 기준 위 
        if x-3>=0:
            up=True
            coors.append(((x-1,y),(x-2,y),(x-3,y)))
        # (x,y) 기준 아래 
        if x+3<self.n_row:
            down=True
            coors.append(((x+1,y),(x+2,y),(x+3,y)))
        # (x,y) 기준 오른쪽 위 
        if right and up:
            coors.append(((x-1,y+1),(x-2,y+2),(x-3,y+3)))
        # (x,y) 기준 오른쪽 아래 
        if right and down:
            coors.append(((x+1,y+1),(x+2,y+2),(x+3,y+3)))
        # (x,y) 기준 왼쪽 위 
        if left and up:
            coors.append(((x-1,y-1),(x-2,y-2),(x-3,y-3)))
        # (x,y) 기준 왼쪽 아래 
        if left and down:
            coors.append(((x+1,y-1),(x+2,y-2),(x+3,y-3)))

        return coors 


    # 승패가 결정됐는지 확인하는 함수
    # 0: not end
    # 1: player 1 win
    # 2: player 2 win
    # 3: draw 
    def check_win(self):

        for x in range(self.n_row-1,-1,-1):
            for y in range(self.n_col):
                piece = self.board[x][y]
                if piece == 0: continue

                coor_list = self.coor_for_8_direction(x,y)
                for coors in coor_list:
                    # print("coors:",coors)
                    if piece == self.board[coors[0]] == self.board[coors[1]] == self.board[coors[2]]:
                        self.win = piece
                        self.done = True
                        return

        if not 0 in self.board[0,:]:
            self.win = 3
        
        if self.win != 0: self.done = True

    def step_human(self, col):
        self.step(col)
        self.print_board()

    # def step_cpu(self):
    #     self.drop_piece(np.random.choice(range(self.n_col)))
    #     self.change_player()
    #     self.print_board()
    #     self.check_win()

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
    def __init__(self, state_size=6*7, action_size=7, gamma=0.999, lr=0.0002, batch_size=256,hidden_size=256, target_update=100, eps=1., memory_len=2000):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.policy_net = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size)
        ) # .to(self.device)
        
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
        
    def select_action(self, state, valid_actions=None):
        if valid_actions is None:
            valid_actions = range(self.action_size)

        if np.random.uniform() < self.eps:
            return np.random.choice(valid_actions)
        with torch.no_grad():
            state = torch.FloatTensor(state)  # .to(self.device)
            q_value = self.policy_net(state)
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

        
        state1_batch = torch.cat([s1 for (s1,a,r,s2,d) in minibatch])
        action_batch = torch.Tensor([a for (s1,a,r,s2,d) in minibatch])
        reward_batch = torch.Tensor([r for (s1,a,r,s2,d) in minibatch])
        state2_batch = torch.cat([s2 for (s1,a,r,s2,d) in minibatch])
        done_batch = torch.Tensor([d for (s1,a,r,s2,d) in minibatch])
        Q1 = self.policy_net(state1_batch) 
        with torch.no_grad():
            Q2 = self.target_net(state2_batch) #B
        
        Y = reward_batch + self.gamma * ((1-done_batch) * torch.max(Q2,dim=1)[0])
        X = Q1.gather(dim=1,index=action_batch.long().unsqueeze(dim=1)).squeeze()
        loss = nn.MSELoss()(X, Y.detach())
        self.losses.append(loss.detach().numpy())
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




    



