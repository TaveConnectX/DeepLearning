import numpy as np
import os
import random
import copy
import math
from collections import deque
import torch
from torch import nn, optim
import torch.nn.functional as F
import time
import env
import nashpy as nash
from env import MCTS, MCTSParallel, SPG
from functions import board_normalization, \
    get_model_config, set_optimizer, \
    get_distinct_actions, is_full_after_my_turn, \
    softmax_policy, get_valid_actions, get_next_board, \
    get_current_time
from ReplayBuffer import RandomReplayBuffer
# models.py 분리 후 이동, 정상 작동하면 지울 듯 
# import torch.nn.init as init
# import torch.nn.functional as F
from models import DQNModel, HeuristicModel, RandomModel, MinimaxModel, \
                    AlphaZeroResNet
import json
import matplotlib.pyplot as plt

# editable hyperparameters
# let iter=10000, then
# learning rate 0.01~0.00001
# batch_size 2^5~2^10
# hidden layer  2^6~2^9
# memory_len  100~100000
# target_update 1~10000


# models = {
#             1:CFLinear,
#             2:CFCNN,
#             3:HeuristicModel,
#             4:RandomModel,
#             5:AlphaZeroResNet,
#             6:ResNetforDQN,
#             7:CNNforMinimax,
#         }


# with open('config.json', 'r') as f:
#     config = json.load(f)

def set_op_agent(agent_name):
    if agent_name == "Heuristic":
        return HeuristicAgent()
    elif agent_name == "Random":
        return ConnectFourRandomAgent()
    elif agent_name == "Minimax":
        return MinimaxAgent()
    elif agent_name == "self":
        return "self"
    else:
        print("invalid op_agent model name")
        exit()

def evaluate_model(agent, record, n_battles=[10,10,10]):
    op_agents = [ConnectFourRandomAgent(), HeuristicAgent(), MinimaxAgent()]
    
    
    w,l,d = env.compare_model(agent, op_agents[0], n_battle=n_battles[0])
    record[0].append(w+d)
    w,l,d = env.compare_model(agent, op_agents[1], n_battle=n_battles[1])
    record[1].append(w+d)
    w,l,d = env.compare_model(agent, op_agents[2], n_battle=n_battles[2])
    record[2].append(w+d)
    


class ConnectFourDQNAgent(nn.Module):
    def __init__(self, state_size=6*7, action_size=7, config_file_name=None, **kwargs):
        super(ConnectFourDQNAgent,self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        config = get_model_config(config_file_name)
        # print(kwargs)
        for key, value in kwargs.items():
            # print("key, value")
            # print(key, value)
            config[key] = value

        self.use_conv=config['use_conv']
        self.use_resnet=config['use_resnet']
        self.use_minimax=config['use_minimax']
        self.use_nash=config['use_nash']
        self.next_state_is_op_state = config['next_state_is_op_state']
        if self.use_minimax and self.next_state_is_op_state:
            print("invalid model structure")
            exit()
        self.policy_net = DQNModel(use_conv=self.use_conv,
                                   use_resnet=self.use_resnet,
                                   use_minimax=self.use_minimax
                                   ).model
        # 실제 업데이트되는 network

        # target network
        self.target_net = copy.deepcopy(self.policy_net)
        # deepcopy하면 파라미터 load를 안해도 되는거 아닌가? 일단 두자
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.lr = config['lr']

        # optimizer는 기본적으로 adam을 사용하겠지만 추후 다른 것으로 실험할 수도 있음
        self.optimizer = set_optimizer(
            config['optimizer'], 
            self.policy_net.parameters(), 
            lr=self.lr
        )
        
        self.selfplay = config['selfplay']
        self.add_pool_freq = config['add_pool_freq']  # no need if no selfplay

        self.memory_len = config['memory_len']
        self.batch_size = config['batch_size']
        # DQN에 사용될 replay memory(buffer)
        self.memory = RandomReplayBuffer(
            buffer_size=self.memory_len,
            batch_size=self.batch_size,
            use_conv=self.use_conv,
            use_minimax=self.use_minimax
        )
        self.gamma = config['gamma']  # discount factor
        self.state_size = state_size
        self.action_size = action_size
        self.target_update = config['target_update']  # target net update 주기(여기선 step)
        self.steps = 0

        self.epi = config['epi']
        self.softmax_const = config['softmax_const'] + np.finfo(np.float32).min
        self.max_temp = config['max_temp']
        self.min_temp = config['min_temp']
        self.temp_decay = (self.min_temp/self.max_temp)**(1/(self.epi*self.softmax_const))
        self.temp = self.max_temp
        self.eps = config['eps']  # DQN에서 사용될 epsilon
        self.noise_while_train = config['noise_while_train']
        
        self.double_dqn = config['double_dqn']


        self.batch_size = config['batch_size']  # size of mini-batch
        self.repeat_reward = config['repeat_reward']  # repeat reward
        self.losses = []  # loss들을 담는 list 
        # record of win rate of random, heuristic, minimax agent
        self.record = [[], [], []]  

    # def get_nash_action(self, q_value,valid_actions, distinct_actions):
    #     # if len(valid_actions) == 1: 
    #     #     return (valid_actions[0],valid_actions[0])
    #     A = q_value.cpu().detach().numpy().reshape(7,7)
    #     A = A[valid_actions][:,valid_actions]
    #     # print(A,B)
    #     game = nash.Game(A)
    #     try:
    #         print(valid_actions)
    #         print(distinct_actions)
    #         print(A)
    #         action_prob1, action_prob2 = game.lemke_howson(initial_dropped_label=0)
    #         if len(action_prob1) != len(valid_actions) or len(action_prob2) != len(valid_actions):        # GAME IS DEGENERATE
    #             equilibria = game.support_enumeration()
    #             action_prob1, action_prob2 = next(equilibria)
    #     except:
    #         equilibria = game.support_enumeration()
    #         action_prob1, action_prob2 = next(equilibria)
    #     # print(valid_actions)
    #     # print(action_prob1)
    #     # print(action_prob2)
    #     # print()
    #     a = np.random.choice (valid_actions, p=action_prob1)
    #     b = np.random.choice (valid_actions, p=action_prob2)
    #     # print("probA:",prob[0])
    #     # print("probB:",prob[1])
    #     if not (a in distinct_actions and a==b):
    #         return (a,b)
    #     else:
    #         return (None, None)

    # https://code.activestate.com/recipes/496825-game-theory-payoff-matrix-solver/
    # 내쉬 균형 찾는 코드 참고 
    def get_nash_prob_and_value(self,payoff_matrix, vas, das, iterations=50):
        if isinstance(payoff_matrix, torch.Tensor):    
            payoff_matrix = payoff_matrix.clone().detach().reshape(7,7)
        elif isinstance(payoff_matrix, np.ndarray):
            payoff_matrix = payoff_matrix.reshape(7,7)
        payoff_matrix = payoff_matrix[vas][:,vas]
        
        '''Return the oddments (mixed strategy ratios) for a given payoff matrix'''
        transpose_payoff = torch.transpose(payoff_matrix,0,1)
        row_cum_payoff = torch.zeros(len(payoff_matrix)).to(self.device)
        col_cum_payoff = torch.zeros(len(transpose_payoff)).to(self.device)

        col_count = np.zeros(len(transpose_payoff))
        row_count = np.zeros(len(payoff_matrix))
        active = 0
        for i in range(iterations):
            row_count[active] += 1 
            col_cum_payoff += payoff_matrix[active]
            active = torch.argmin(col_cum_payoff)
            col_count[active] += 1 
            row_cum_payoff += transpose_payoff[active]
            active = torch.argmax(row_cum_payoff)
            
        value_of_game = (max(row_cum_payoff) + min(col_cum_payoff)) / 2.0 / iterations  
        row_prob = row_count / iterations
        col_prob = col_count / iterations
        
        return row_prob, col_prob, value_of_game
    
    def get_nash_action(self, q_value,valid_actions, distinct_actions):
        # if len(valid_actions) == 1: 
        #     return (valid_actions[0],valid_actions[0])
        # print(A,B)
        action_prob1, action_prob2, value = self.get_nash_prob_and_value(q_value, valid_actions, distinct_actions)
        # print(valid_actions)
        # print(action_prob1)
        # print(action_prob2)
        # print()
        a = np.random.choice (valid_actions, p=action_prob1)
        b = np.random.choice (valid_actions, p=action_prob2)
        # print("probA:",prob[0])
        # print("probB:",prob[1])
        if not (a in distinct_actions and a==b):
            return (a,b)
        else:
            return (None, None)

    # softmax를 포함한 minimax action sampling
    def get_minimax_action(self, q_value,valid_actions, distinct_actions, temp=0):
        if is_full_after_my_turn(valid_actions, distinct_actions):
                return (valid_actions[0], np.random.choice(valid_actions))
        
        if self.use_nash:
            a,b = self.get_nash_action(q_value,valid_actions, distinct_actions)
            if (a,b) != (None, None):
                return a,b
        
            
        q_dict = {}
        # print(valid_actions)
        # print(distinct_actions)
        for a in valid_actions:
            q_dict[a] = []
            for b in valid_actions:
                if a in distinct_actions and a==b: continue
                idx = 7*a + b
                # print(a,b)
                # print(q_value[idx])
                # print(q_dict[a][1])
                
                q_dict[a].append((b, -q_value[idx]))
            
            op_action, value = softmax_policy(torch.tensor(q_dict[a]), temp=temp)
            # if torch.isnan(value):
            #     print(a,b)
            #     print(q_value.reshape(7,7))
            #     print(q_dict)
            q_dict[a] = (op_action, -1 * value)

        qs_my_turn = [[key, value[1]] for key, value in q_dict.items()]
        action, value = softmax_policy(torch.tensor(qs_my_turn), temp=temp)
        # if torch.isnan(value):
        #         print(a,b)
        #         print(q_value.reshape(7,7))
        #         print(q_dict)


        return (action, q_dict[action][0])
    
    def select_action(self, state, env, player=1):
        
        valid_actions = env.valid_actions

        if self.use_minimax:
            distinct_actions = get_distinct_actions(env)
            
            if is_full_after_my_turn(valid_actions, distinct_actions):
                return (valid_actions[0], np.random.choice(valid_actions))
            if np.max([np.random.uniform(), self.softmax_const]) < self.eps:
                while True:
                    a,b = np.random.choice(valid_actions), np.random.choice(valid_actions)
                    if a==b and a in distinct_actions:continue
                    else: break
                        
                return (a,b)
            
        
        else:
            if np.max([np.random.uniform(), self.softmax_const]) < self.eps:
                return np.random.choice(valid_actions)
            
        with torch.no_grad():
            state_ = torch.FloatTensor(state).to(self.device)
            # CNN일 때만 차원을 바꿔줌 
            if self.use_conv:
                state_ = state_.reshape(6,7)
                state_ = state_.unsqueeze(0).unsqueeze(0)  # (6,7) -> (1,1,6,7)
            else: state_ = state_.flatten()

            q_value = self.policy_net(state_)

            # print(env.board)
            # print(((q_value.reshape(7,7)*100).int()/100.).v)

            if True in torch.isnan(q_value):
                print( q_value.reshape(7,7))
                print(state)
                print(state_)
                print(env.board)
                exit()
                print()
            # temp=0 은 greedy action을 의미하므로
            temp = 0 if self.softmax_const < self.eps else self.temp

            if self.use_minimax:
                # print("state:",state)
                # print("valid_actions:",valid_actions)
                # print("q_value:",q_value)
                
                a,b = self.get_minimax_action(
                    q_value.squeeze(0),
                    valid_actions, 
                    distinct_actions,
                    temp=temp
                )
                
                # for debugging
                # print(q_value.reshape(7,7))
                # print(valid_actions)
                # print(distinct_actions)
                # print(temp)
                # print(a,b)
                return (a, b)
            
            else:
                # print("state:",state)
                # print("valid_actions:",valid_actions)
                # print("q_value:",q_value)
                
                valid_q_values = q_value.squeeze()[torch.tensor(valid_actions)].to(self.device)
                valid_index_q_values = torch.stack([torch.tensor(valid_actions).to(self.device), valid_q_values], dim=0)
                
                valid_index_q_values = torch.transpose(valid_index_q_values,0,1)
                action, value = softmax_policy(valid_index_q_values,temp=temp)

                return action

                # return valid_actions[torch.argmax(valid_q_values)]
        
    # # replay buffer에 경험 추가 
    # def append_memory(self, state, action, reward, next_state, done):
    #     if self.policy_net.model_type == 'Linear':
    #         self.memory.append((state.flatten(), action, reward, next_state.flatten(), done))
    #     else: 
    #         self.memory.append((state.reshape(6,7), action, reward, next_state.reshape(6,7), done))
            

    def collect_data(self, env, op_model):

        players = {1:self, 2:op_model}

        while self.memory.get_length() < self.memory.start_size:
            print("num collected:",self.memory.get_length())
            # if self.memory.get_length() > 20:
            #     s = torch.round(self.memory.buffer[-2][0]).reshape(6,7)
            #     a = self.memory.buffer[-2][1]
            #     b = self.memory.buffer[-2][2]
            #     r = self.memory.buffer[-2][3]
            #     s_prime = torch.round(self.memory.buffer[-2][4]).reshape(6,7)
            #     m = self.memory.buffer[-2][5]
            #     d = self.memory.buffer[-2][6]
            #     print(s,a,b,r,s_prime,m,d)
            #     print()
            #     s = torch.round(self.memory.buffer[-1][0]).reshape(6,7)
            #     a = self.memory.buffer[-1][1]
            #     b = self.memory.buffer[-1][2]
            #     r = self.memory.buffer[-1][3]
            #     s_prime = torch.round(self.memory.buffer[-1][4]).reshape(6,7)
            #     m = self.memory.buffer[-1][5]
            #     d = self.memory.buffer[-1][6]
            #     print(s,a,b,r,s_prime,m,d)
            #     print()


            env.reset()

            state_ = board_normalization(noise=self.noise_while_train, env=env, use_conv=players[env.player].use_conv)
            state = torch.from_numpy(state_).float()
            done = False

            past_state, past_action, past_reward, past_done = state, None, None, done
            
            while not done:
                # 원래는 player, op_player 였지만, 직관적인 이해를 위해 수정 
                turn = env.player
                op_turn = 2//turn
                
                action = players[turn].select_action(state, env, player=turn)
                
                if players[turn].use_minimax:
                    op_action_prediction = action[1]
                    action = action[0]
                
                observation, reward, done = env.step(action)
                if self.use_minimax and self.use_nash:
                    mask = torch.zeros(7)
                    VA = get_valid_actions(observation)
                    DA = get_distinct_actions(env)
                    mask[VA] = 1
                    mask[DA] = 2
                elif self.use_minimax:
                    mask = torch.ones(7,7)
                    VA = get_valid_actions(observation)
                    DA = get_distinct_actions(env)
                    for a in range(7):
                        if not a in VA:
                            mask[a,:] = 0
                            mask[:,a] = 0
                    for da in DA:
                        mask[da,da] = 0
                    # for debugging
                    # print(observation)
                    # print(VA)
                    # print(DA)
                    # print(mask)
                    # print()
                else:
                    mask = torch.zeros(7)
                    VA = get_valid_actions(observation)
                    for va in VA:
                        mask[va] = 1

                    
                        
                op_state_ = board_normalization(noise=self.noise_while_train, env=env, use_conv=players[turn].use_conv)
                op_state = torch.from_numpy(op_state_).float() 

                if past_action is not None:  # 맨 처음이 아닐 때 
                    # 경기가 끝났을 때(중요한 경험)
                    if done:
                        repeat = 1
                        # 중요한 경험일 때는 더 많이 memory에 추가해준다(optional)
                        if reward > 0: repeat = self.repeat_reward
                        for _ in range(repeat):
                            # 돌을 놓자마자 끝났으므로, next_state가 반전됨, 따라서 -1을 곱해준다
                            
                            if turn==1:
                                if self.use_minimax:
                                    self.memory.add(state, action, op_action_prediction, reward, op_state*-1,mask,done)
                                elif self.next_state_is_op_state:
                                    self.memory.add(state, action, reward, op_state,mask,done)
                                else:
                                    self.memory.add(state,action, reward, op_state*-1, mask,done)
                                
                            if turn==2:
                                if self.use_minimax:
                                    self.memory.add(past_state, past_action, action, -reward, op_state, mask,done)
                                elif self.next_state_is_op_state:
                                    
                                    self.memory.add(past_state, past_action,past_reward, state,mask, past_done)
                                else:
                                    self.memory.add(past_state, past_action, -reward, op_state, mask,done)
                                


                    # 경기가 끝나지 않았다면
                    elif turn==2:  # 내 경험만 수집한다
                        if self.use_minimax:
                            self.memory.add(past_state, past_action, action, past_reward, op_state, mask,past_done)
                        elif self.next_state_is_op_state:
                            self.memory.add(past_state, past_action, past_reward, state,mask,past_done)
                        else:
                            self.memory.add(past_state, past_action, past_reward, op_state, mask,past_done)
                        

                # info들 업데이트 해줌 
                past_state = state
                past_action = action
                past_reward = reward
                past_done = done
                state = op_state
                
                # 게임이 끝났다면 나가기 
                if done: break



    def minimax_train(self,epi,env):
        env.reset()
        print(self.eps, epi)
        self.collect_data(env, self)


        if self.selfplay:
            players = {1: self}
            new_model = copy.deepcopy(self)
            new_model.policy_net.eval()
            new_model.target_net.eval()
            new_model.eps = 0.1
            
            pool = deque([new_model], maxlen=1000)
        # models 딕셔너리는 전역 변수로 사용하므로, players로 변경 
        else: players = {1: self, 2: self}

        for i in range(epi):
            if self.selfplay: 
                random.shuffle(pool)
                players[2] = random.choice(pool)
            if i!=0 and i%200==0: 
                env.print_board(clear_board=False)
                print("epi:",i, ", agent's step:",self.steps)
                # 얼마나 학습이 진행되었는지 확인하기 위해, 모델 성능 측정 
                evaluate_model(self, record=self.record)
                
                print(self.record[0][-1],self.record[1][-1], self.record[2][-1])
                # print("agent의 승률이 {}%".format(int(100*record[0]/sum(record))))
                print("loss:",sum(self.losses[-101:-1])/100.)
                if self.eps > self.softmax_const: print("epsilon:",self.eps)
                else: print("temp:",self.temp)
                # simulate_model() 을 실행시키면 직접 행동 관찰 가능
                # simulate_model(self, op_model)
            
            env.reset()

            if env.player == 2:
                action = random.randint(0,6)
                env.step(action)
            state_ = board_normalization(noise=self.noise_while_train, env=env, use_conv=self.use_conv)
            state = torch.from_numpy(state_).float()
            done = False

            
            while not done:
                # 원래는 player, op_player 였지만, 직관적인 이해를 위해 수정 
                turn = env.player
                if turn != 1:
                    print("this cannot be happen in minimax")
                    exit()
                
                action, op_action = players[turn].select_action(state, env, player=turn)
                
                
                
                observation, reward, done = env.step(action)
                if self.use_nash:
                    mask = torch.zeros(7)
                    VA = get_valid_actions(observation)
                    DA = get_distinct_actions(env)
                    mask[VA] = 1
                    mask[DA] = 2
                else:
                    mask = torch.ones(7,7)
                    VA = get_valid_actions(observation)
                    DA = get_distinct_actions(env)
                    for a in range(7):
                        if not a in VA:
                            mask[a,:] = 0
                            mask[:,a] = 0
                    for da in DA:
                        mask[da,da] = 0
                # for debugging
                # print(observation)
                # print(VA)
                # print(DA)
                # print(mask)
                # print()
                op_state_ = board_normalization(noise=self.noise_while_train, env=env, use_conv=self.use_conv)
                op_state = torch.from_numpy(op_state_).float() 

                # 경기가 끝났을 때(중요한 경험)
                if done:
                    # 중요한 경험일 때는 더 많이 memory에 추가해준다(optional)
                    if reward > 0: repeat = self.repeat_reward
                    for _ in range(repeat):
                        # 돌을 놓자마자 끝났으므로, next_state가 반전됨, 따라서 -1을 곱해준다
                        self.memory.add(state, action, op_action, reward, op_state*-1,mask,done)
                        # print("s:\n",torch.round(state).reshape(6,7).int())
                        # print("a:\n", action)
                        # print("b:\n", op_action)
                        # print("r:\n", reward)
                        # print("s_prime\n", torch.round(op_state*-1).reshape(6,7).int())
                        # print("m:\n", mask.reshape(7,7).int())
                        # print("d:\n", done)

                    break
                            
                # 아직 안끝났으면,
                if self.selfplay:
                    op_real_action, _ = players[2].select_action(op_state, env, player=turn)
                    op_observation, op_reward, op_done = env.step(op_real_action) 
                else:
                    op_observation, op_reward, op_done = env.step(op_action)

                done = done or op_done
                if self.use_nash:
                    mask = torch.zeros(7)
                    VA = get_valid_actions(observation)
                    DA = get_distinct_actions(env)
                    mask[VA] = 1
                    mask[DA] = 2
                else:
                    mask = torch.ones(7,7)
                    VA = get_valid_actions(op_observation)
                    DA = get_distinct_actions(env)
                    for a in range(7):
                        if not a in VA:
                            mask[a,:] = 0
                            mask[:,a] = 0
                    for da in DA:
                        mask[da,da] = 0   

                next_state_ = board_normalization(noise=self.noise_while_train, env=env, use_conv=players[turn].use_conv)
                next_state = torch.from_numpy(next_state_).float() 
                  
                
                self.memory.add(state, action, op_action, reward-op_reward, next_state, mask,done)
                # print("s:\n",torch.round(state).reshape(6,7).int())
                # print("a:\n", action)
                # print("b:\n", op_action)
                # print("r:\n", reward-op_reward)
                # print("s_prime\n", torch.round(next_state).reshape(6,7).int())
                # print("m:\n", mask.reshape(7,7).int())
                # print("d:\n", done)

                            
                         

                # info들 업데이트 해줌 
                state = next_state.clone()
                
                # replay buffer 를 이용하여 mini-batch 학습
                if self.use_minimax and self.use_nash:
                    self.replay_nash()
                else:
                    self.replay()

                if self.selfplay and self.steps and not self.steps%self.add_pool_freq:
                    print("added in pool")
                    new_model = copy.deepcopy(self)
                    new_model.policy_net.eval()
                    new_model.target_net.eval()
                    new_model.eps = 0.1
                    new_model.softmax_const = 0
                    new_model.temp = 0
                    pool.append(new_model)
                # if Qagent.memory and abs(Qagent.memory[-1][2])!=1:
                #     print("state:\n",torch.round(Qagent.memory[-1][0]).int())
                #     print("action:",Qagent.memory[-1][1])
                #     print("reward:",Qagent.memory[-1][2])
                #     print("next_state\n",torch.round(Qagent.memory[-1][3]).int())
            
            if self.eps > 0.1: self.eps -= (1/epi)
            if self.eps < self.softmax_const:
                self.temp *= self.temp_decay

            

    def train(self,epi,env:env.ConnectFourEnv,op_model):
        if self.use_minimax:
            self.minimax_train(epi,env)
            return
        env.reset()

        # self.eps += self.memory.get_maxlen()//10/epi
        # epi += self.memory.get_maxlen()//10
        print(self.eps, epi)
        if self.selfplay:
            players = {1: self}
            new_model = copy.deepcopy(self)
            new_model.policy_net.eval()
            new_model.target_net.eval()
            new_model.eps = 0.1
            
            pool = deque([new_model], maxlen=200)
        # models 딕셔너리는 전역 변수로 사용하므로, players로 변경 
        else: players = {1: self, 2: op_model}


        self.collect_data(env, self if self.selfplay else op_model)
        
        for i in range(epi):
            
            if self.selfplay: 
                random.shuffle(pool)
                players[2] = random.choice(pool)
            # 100번마다 loss, eps 등의 정보 표시
            if i!=0 and i%200==0: 
                #env.print_board(clear_board=False)
                print("epi:",i, ", agent's step:",self.steps)
                # 얼마나 학습이 진행되었는지 확인하기 위해, 모델 성능 측정 
                # evaluate_model(self, record=self.record)
                
                # print(self.record[0][-1],self.record[1][-1], self.record[2][-1])
                # print("agent의 승률이 {}%".format(int(100*record[0]/sum(record))))
                print("loss:",sum(self.losses[-101:-1])/100.)
                if self.eps > self.softmax_const: print("epsilon:",self.eps)
                else: print("temp:",self.temp)
                # simulate_model() 을 실행시키면 직접 행동 관찰 가능
                # simulate_model(self, op_model)

            
            env.reset()

            state_ = board_normalization(noise=self.noise_while_train, env=env, use_conv=players[env.player].use_conv)
            state = torch.from_numpy(state_).float()
            done = False

            past_state, past_action, past_reward, past_done = state, None, None, done
            
            while not done:
                # 원래는 player, op_player 였지만, 직관적인 이해를 위해 수정 
                turn = env.player
                op_turn = 2//turn
                
                action = players[turn].select_action(state, env, player=turn)
                
                if players[turn].use_minimax:
                    op_action_prediction = action[1]
                    action = action[0]
                
                observation, reward, done = env.step(action)
                if self.use_minimax:
                    mask = torch.ones(7,7)
                    VA = get_valid_actions(observation)
                    DA = get_distinct_actions(env)
                    for a in range(7):
                        if not a in VA:
                            mask[a,:] = 0
                            mask[:,a] = 0
                    for da in DA:
                        mask[da,da] = 0
                    # for debugging
                    # print(observation)
                    # print(VA)
                    # print(DA)
                    # print(mask)
                    # print()
                else:
                    mask = torch.zeros(7)
                    VA = get_valid_actions(observation)
                    for va in VA:
                        mask[va] = 1

                    
                        
                op_state_ = board_normalization(noise=self.noise_while_train, env=env, use_conv=players[turn].use_conv)
                op_state = torch.from_numpy(op_state_).float() 

                if past_action is not None:  # 맨 처음이 아닐 때 
                    # 경기가 끝났을 때(중요한 경험)
                    if done:
                        repeat = 1
                        # 중요한 경험일 때는 더 많이 memory에 추가해준다(optional)
                        if reward > 0: repeat = self.repeat_reward
                        for _ in range(repeat):
                            # 돌을 놓자마자 끝났으므로, next_state가 반전됨, 따라서 -1을 곱해준다
                            
                            if turn==1:
                                if self.use_minimax:
                                    self.memory.add(state, action, op_action_prediction, reward, op_state*-1,mask,done)
                                elif self.next_state_is_op_state:
                                    self.memory.add(state, action, reward, op_state,mask,done)
                                else:
                                    self.memory.add(state,action, reward, op_state*-1, mask,done)
                                # print for debugging
                                # print("for player")
                                # print("state:\n",torch.round(state).reshape(6,7).int())
                                # print("action:",action)
                                
                                # print("reward:",reward)
                                # print("next_state:\n",torch.round(op_state*-1).reshape(6,7).int())
                                # print("mask:\n",torch.round(mask))
                                # print()
                            # 내가 이겼으므로 상대는 음의 보상을 받음 
                            #Qmodels[op_player].append_memory(past_state, past_action, -reward, op_state, done)
                            if turn==2:
                                if self.use_minimax:
                                    self.memory.add(past_state, past_action, action, -reward, op_state, mask,done)
                                elif self.next_state_is_op_state:
                                    
                                    self.memory.add(past_state, past_action,past_reward, state,mask, past_done)
                                else:
                                    self.memory.add(past_state, past_action, -reward, op_state, mask,done)
                                # print for debugging
                                # print("for opponent")
                                # print("state:\n",torch.round(past_state).reshape(6,7).int())
                                # print("action:",past_action)
                                # print("op_action:",action)
                                # print("reward:",-reward)
                                # print("next_state\n",torch.round(state).reshape(6,7).int())
                                # print("next_state\n",torch.round(op_state).reshape(6,7).int())
                                # print("mask\n",mask)
                                # print()


                    # 경기가 끝나지 않았다면
                    elif turn==2:  # 내 경험만 수집한다
                        if self.use_minimax:
                            self.memory.add(past_state, past_action, action, past_reward, op_state, mask,past_done)
                        elif self.next_state_is_op_state:
                            self.memory.add(past_state, past_action, past_reward, state,mask,past_done)
                        else:
                            self.memory.add(past_state, past_action, past_reward, op_state, mask,past_done)
                        # print for debugging
                        # print("for opponent")
                        # print("state:\n",torch.round(past_state).reshape(6,7).int())
                        # print("action:",past_action)
                        # print("reward:",past_reward)
                        # print("next_state\n",torch.round(state*-1).reshape(6,7).int())
                        # print("mask\n",mask)
                        # print()

                
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
                if self.selfplay and self.steps and not self.steps%self.add_pool_freq:
                    print("added in pool")
                    new_model = copy.deepcopy(self)
                    new_model.policy_net.eval()
                    new_model.target_net.eval()
                    new_model.eps = 0.1
                    new_model.softmax_const = 0
                    new_model.temp = 0
                    pool.append(new_model)
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
            if self.eps < self.softmax_const:
                self.temp *= self.temp_decay


    # def train_selfplay(self, epi, env, pool, add_pool):
    #     env.reset()

    #     # models 딕셔너리는 전역 변수로 사용하므로, players로 변경 
    #     players = {1: self}

    #     for i in range(epi):

    #         players[2] = random.choice(pool)
    #         # 100번마다 loss, eps 등의 정보 표시
    #         if i!=0 and i%100==0: 
    #             env.print_board(clear_board=False)
    #             print("epi:",i, ", agent's step:",self.steps)
    #             # 얼마나 학습이 진행되었는지 확인하기 위해, 모델 성능 측정 
    #             record = compare_model(self, players[2], n_battle=100)
    #             print(record)
    #             print("agent의 승률이 {}%".format(int(100*record[0]/sum(record))))
    #             print("loss:",sum(self.losses[-101:-1])/100.)
    #             print("epsilon:",self.eps)
    #             # simulate_model() 을 실행시키면 직접 행동 관찰 가능
    #             # simulate_model(self, op_model)

            
    #         env.reset()

    #         state_ = board_normalization(noise=self.noise_while_train, env=env, use_conv=players[env.player].use_conv)
    #         state = torch.from_numpy(state_).float()
    #         done = False

    #         past_state, past_action, past_reward, past_done = state, None, None, done
            
    #         while not done:
    #             # 원래는 player, op_player 였지만, 직관적인 이해를 위해 수정 
    #             turn = env.player
    #             op_turn = 2//turn
                
    #             action = players[turn].select_action(state, env, player=turn)
    #             if self.use_minimax:
    #                 op_action_prediction = action[1]
    #                 action = action[0]



    #             observation, reward, done = env.step(action)
    #             op_state_ = board_normalization(noise=self.noise_while_train, env=env, use_conv=players[turn].use_conv)
    #             op_state = torch.from_numpy(op_state_).float() 

    #             if past_action is not None:  # 맨 처음이 아닐 때 
    #                 # 경기가 끝났을 때(중요한 경험)
    #                 if done:
    #                     repeat = 1
    #                     # 중요한 경험일 때는 더 많이 memory에 추가해준다(optional)
    #                     if reward > 0: repeat = self.repeat_reward
    #                     for j in range(repeat):
    #                         # 돌을 놓자마자 끝났으므로, next_state가 반전됨, 따라서 -1을 곱해준다
    #                         if turn==1:
    #                             if self.use_minimax:
    #                                 self.append_memory(state,action,op_action_prediction, reward, op_state*-1, done)
    #                             else:
    #                                 self.append_memory(state,action, reward, op_state*-1, done)
    #                             # print for debugging
    #                             # print("for player")
    #                             # print("state:\n",torch.round(state).reshape(6,7).int())
    #                             # print("action:",action)
    #                             # print("reward:",reward)
    #                             # print("next_state\n",torch.round(op_state*-1).reshape(6,7).int())
    #                             # print()
    #                         # 내가 이겼으므로 상대는 음의 보상을 받음 
    #                         #Qmodels[op_player].append_memory(past_state, past_action, -reward, op_state, done)
    #                         if turn==2:
    #                             if self.use_minimax:
    #                                 self.append_memory(past_state, past_action, action, -reward, op_state, done)
    #                             else:
    #                                 self.append_memory(past_state, past_action, -reward, op_state, done)
    #                             # print for debugging
    #                             # print("for opponent")
    #                             # print("state:\n",torch.round(past_state).reshape(6,7).int())
    #                             # print("action:",past_action)
    #                             # print("reward:",-reward)
    #                             # print("next_state\n",torch.round(op_state).reshape(6,7).int())
    #                             # print()


    #                 # 경기가 끝나지 않았다면
    #                 elif turn==2:  # 내 경험만 수집한다
    #                     if self.use_minimax:
    #                         self.append_memory(past_state, past_action, action, past_reward, op_state, past_done)
    #                     else:
    #                         self.append_memory(past_state, past_action, past_reward, op_state, past_done)
    #                     # print for debugging
    #                     # print("for opponent")
    #                     # print("state:\n",torch.round(past_state).reshape(6,7).int())
    #                     # print("action:",past_action)
    #                     # print("reward:",past_reward)
    #                     # print("next_state\n",torch.round(op_state).reshape(6,7).int())
    #                     # print()


    #             # info들 업데이트 해줌 
    #             past_state = state
    #             past_action = action
    #             past_reward = reward
    #             past_done = done
    #             state = op_state
                
    #             # replay buffer 를 이용하여 mini-batch 학습
    #             self.replay()
    #             if self.steps%add_pool == 0:
    #                 print("added in pool")
    #                 new_model = copy.deepcopy(self)
    #                 new_model.policy_net.eval()
    #                 new_model.target_net.eval()
    #                 new_model.eps = 0
    #                 pool.append(new_model)
    #             # if Qagent.memory and abs(Qagent.memory[-1][2])!=1:
    #             #     print("state:\n",torch.round(Qagent.memory[-1][0]).int())
    #             #     print("action:",Qagent.memory[-1][1])
    #             #     print("reward:",Qagent.memory[-1][2])
    #             #     print("next_state\n",torch.round(Qagent.memory[-1][3]).int())
                
    #             # 게임이 끝났다면 나가기 
    #             if done: break
    #         # print("eps:",Qagent.eps)
            
    #         # epsilon-greedy
    #         # min epsilon을 가지기 전까지 episode마다 조금씩 낮춰준다(1 -> 0.1)
    #         if self.eps > 0.1: self.eps -= (1/epi)

    def replay_nash(self):
        if self.memory.get_length() < self.memory.start_size:
            return
        
        # 메모리를 계속 섞어
        # self.memory.shuffle()

        
        s_batch, a_batch, r_batch, s_prime_batch, m_batch, d_batch = self.memory.sample()

        # print("state1_batch:",state1_batch.shape)
        Q1 = self.policy_net(s_batch)  # (256,7)
        with torch.no_grad():
            Q2 = self.target_net(s_prime_batch)
            #m_batch = m_batch.reshape(-1,1)
            
        nash_q_values = []
        
        # target Q value들 
        if self.double_dqn:

            mask_qs = self.policy_net(s_prime_batch).reshape(-1,7,7)
            Q2 = Q2.reshape(-1,7,7)
            for i in range(mask_qs.shape[0]):
                mask_q = mask_qs[i]
                valid_as = torch.nonzero(m_batch[i], as_tuple=True)[0].cpu().numpy()
                distinct_as = torch.nonzero(m_batch[i]==2, as_tuple=True)[0].cpu().numpy()
                if not valid_as.tolist():
                    q = 0

                else:
                    a_prob, b_prob, _ = self.get_nash_prob_and_value(mask_q, valid_as, distinct_as)
                    norm_a_prob, norm_b_prob = np.zeros(7), np.zeros(7)
                    norm_a_prob[valid_as] = a_prob
                    norm_b_prob[valid_as] = b_prob
                    q = np.sum(np.multiply.outer(norm_a_prob, norm_b_prob) * Q2[i].cpu().numpy())
                
                nash_q_values.append(q)

            nash_q_values = torch.Tensor(nash_q_values).to(self.device).reshape(-1)
                
            Y = r_batch + self.gamma * ((1-d_batch) * nash_q_values)

        # only DQN
        else:
            mask_qs = Q2.reshape(-1,7,7)
            for i in range(Q2.shape[0]):
                mask_q = mask_qs[i]
                valid_as = torch.nonzero(m_batch[i], as_tuple=True)[0].cpu().numpy()
                distinct_as = torch.nonzero(m_batch[i]==2, as_tuple=True)[0].cpu().numpy()
                if not valid_as.tolist():
                    q = 0
                else:
                    a_prob, b_prob, q = self.get_nash_prob_and_value(mask_q, valid_as, distinct_as)
                
                nash_q_values.append(q)
            
            nash_q_values = torch.Tensor(nash_q_values).to(self.device).reshape(-1)
            # print("r size:",r_batch.shape)
            # print("nash_q_values size:",nash_q_values.shape)
            Y = r_batch + self.gamma * ((1-d_batch) * nash_q_values)
            

        
        # 해당하는 action을 취한 q value들
        X = Q1.gather(dim=1,index=a_batch.long().unsqueeze(dim=1)).squeeze()
        # print(X)
        # print(X.shape)
        # print(Y)
        # print(Y.shape)
        # if r_batch[20] == 0:
        #     print(Q1[20].reshape(7,7))
        #     print(torch.round(s_batch[20]))
        #     print(torch.round(s_prime_batch[20]))
        #     print(mask_q[20])
        #     print(a_batch[20]//7, a_batch[20]%7)
        #     print(X[20])
        #     print(Y[20])
        #     print()



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
        # loss에 nan이 있다면 처리
        # if True in torch.isnan(loss):
        #     idx = torch.nonzero(torch.isnan(Y)).squeeze()
        #     print(idx)
        #     print(X[idx])
        #     print(Y[idx])
        #     print(loss)

        #     print(mask_q[idx])
        #     print(Q2[idx])
        #     print(Q1[idx])
        #     print()
        #     print(s_batch[idx])
        #     print(a_batch[idx])
        #     print(r_batch[idx])
        #     print(m_batch[idx])
        #     print(s_prime_batch[idx])
        #     print(d_batch[idx])
        #     exit()
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


    
    # mini-batch로 업데이트 
    def replay(self):
        # if self.memory.get_length() < self.memory.get_maxlen():
        #     return
        if self.memory.get_length() < self.memory.start_size:
            return
        
        # 메모리를 계속 섞어
        # self.memory.shuffle()

        # batch size 만큼 랜덤으로 꺼낸다 
        # minibatch = random.sample(self.memory, self.batch_size)


        # if self.use_conv:
        #     # state_batch.shape: (batch_size, 1, 6, 7)
        #     state1_batch = torch.stack([s1 for (s1,a,r,s2,d) in minibatch]).unsqueeze(1).to(self.device)
        #     state2_batch = torch.stack([s2 for (s1,a,r,s2,d) in minibatch]).unsqueeze(1).to(self.device)
            
        # else:
        #     # state_batch.shape: (batch_size, 42)
            
        #     state1_batch = torch.stack([s1 for (s1,a,r,s2,d) in minibatch]).to(self.device)
        #     state2_batch = torch.stack([s2 for (s1,a,r,s2,d) in minibatch]).to(self.device)

        # # action_batch.shape: (batch_size, )
        # action_batch = torch.Tensor([a for (s1,a,r,s2,d) in minibatch]).to(self.device)
        # reward_batch = torch.Tensor([r for (s1,a,r,s2,d) in minibatch]).to(self.device)
        # done_batch = torch.Tensor([d for (s1,a,r,s2,d) in minibatch]).to(self.device)
        
        s_batch, a_batch, r_batch, s_prime_batch, m_batch, d_batch = self.memory.sample()

        
        
        # print("state1_batch:",state1_batch.shape)
        Q1 = self.policy_net(s_batch)  # (256,7)
        with torch.no_grad():
            Q2 = self.target_net(s_prime_batch)
        m_batch = m_batch.bool()
        NSIOP = -1 if self.next_state_is_op_state else 1
        # target Q value들 
        
    


        if self.use_minimax and self.double_dqn:
            mask_q = self.policy_net(s_prime_batch).reshape(-1,7,7) * m_batch
            mask_q[mask_q==0] = float('inf')
            op_qs, op_idxs = torch.min(mask_q, dim=2)
            op_qs = torch.nan_to_num(op_qs, posinf=float('-inf'))
            qs, idxs = torch.max(op_qs, dim=1)
            op_idxs = torch.gather(op_idxs ,1, idxs.reshape(-1,1)).squeeze()
            acts = idxs*7 + op_idxs
            
            Q2 = Q2.gather(1, acts.unsqueeze(dim=1)).squeeze()
            Y = r_batch + self.gamma * ((1-d_batch) * Q2)



        elif self.use_minimax:
            mask_q = Q2.reshape(-1,7,7) * m_batch
            mask_q[mask_q==0] = float('inf')
            op_qs = torch.amin(mask_q, dim=2)
            op_qs = torch.nan_to_num(op_qs, posinf=float('-inf'))
            qs = torch.nan_to_num(torch.amax(op_qs, dim=1), neginf=0.)
            Y = r_batch + self.gamma * ((1-d_batch) * qs)
            

        elif self.double_dqn:
            mask_q = self.policy_net(s_prime_batch) * m_batch
            mask_q[mask_q==0] = float('-inf')
            Q2 = Q2.gather(1, mask_q.argmax(dim=1).unsqueeze(dim=1)).squeeze()
            Y = r_batch + self.gamma * NSIOP*((1-d_batch)* Q2)

        else:
            mask_q = Q2 * m_batch
            mask_q[mask_q==0] =float('-inf')
            Y = r_batch + self.gamma * NSIOP*((1-d_batch) * torch.amax(mask_q,dim=1))
        
        # 해당하는 action을 취한 q value들
        X = Q1.gather(dim=1,index=a_batch.long().unsqueeze(dim=1)).squeeze()
       
        # if r_batch[20] == 0:
        #     print(Q1[20].reshape(7,7))
        #     print(torch.round(s_batch[20]))
        #     print(torch.round(s_prime_batch[20]))
        #     print(mask_q[20])
        #     print(a_batch[20]//7, a_batch[20]%7)
        #     print(X[20])
        #     print(Y[20])
        #     print()



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
        # loss에 nan이 있다면 처리
        # if True in torch.isnan(loss):
        #     idx = torch.nonzero(torch.isnan(Y)).squeeze()
        #     print(idx)
        #     print(X[idx])
        #     print(Y[idx])
        #     print(loss)

        #     print(mask_q[idx])
        #     print(Q2[idx])
        #     print(Q1[idx])
        #     print()
        #     print(s_batch[idx])
        #     print(a_batch[idx])
        #     print(r_batch[idx])
        #     print(m_batch[idx])
        #     print(s_prime_batch[idx])
        #     print(d_batch[idx])
        #     exit()
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



# DQNAgent에 통합해서 주석처리 
# DQN with minimax
# class MinimaxDQNAgent(nn.Module):
#     def __init__(self, state_size=6*7, action_size=7, gamma=0.99, lr=0.0001, batch_size=64, target_update=20, eps=1., memory_len=6400,repeat_reward=2,model_num=7):
#         super(MinimaxDQNAgent,self).__init__()
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
#         self.policy_net = models[model_num]()
#         self.target_net = copy.deepcopy(self.policy_net)
#         # deepcopy하면 파라미터 load를 안해도 되는거 아닌가? 일단 두자
#         self.target_net.load_state_dict(self.policy_net.state_dict())
#         self.target_net.eval()


#         self.lr = lr
#         # optimizer는 기본적으로 adam을 사용하겠지만 추후 다른 것으로 실험할 수도 있음
#         self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        
#         self.memory_len = memory_len
#         # DQN에 사용될 replay memory(buffer)
#         self.memory = deque(maxlen=self.memory_len)
#         self.gamma = gamma  # discount factor
#         self.state_size = state_size
#         self.action_size = action_size
#         self.target_update = target_update  # target net update 주기(여기선 step)
#         self.steps = 0
#         self.eps = eps  # DQN에서 사용될 epsilon
#         self.batch_size = batch_size  # size of mini-batch
#         self.repeat_reward = repeat_reward
#         self.losses = []  # loss들을 담는 list 

   


#     # 한 칸만 남았으면 pair 액션이 불가능하므로 체크가 필요 
#     def is_full_after_my_turn(self, valid_actions, distinct_actions):
#         if len(valid_actions)==1 and len(distinct_actions)==1:
#             return True
#         else: return False

#     def get_minimax_action(self, q_value,valid_actions, distinct_actions):
#         q_dict = {}
#         for a in valid_actions:
#             q_dict[a] = (None, np.inf)
#             for b in valid_actions:
#                 if a in distinct_actions and a==b: continue
#                 idx = 7*a + b
#                 # print(a,b)
#                 # print(q_value[idx])
#                 # print(q_dict[a][1])
#                 if q_value[idx] <= q_dict[a][1]:
#                     q_dict[a] = (b, q_value[idx])

#         max_key = None
#         max_value = float('-inf')
#         for a, (b, q) in q_dict.items():
#             if q > max_value:
#                 max_key = a
#                 max_value = q

#         return (max_key, q_dict[max_key][0])
    
#     def select_action(self, state, env, player=1):
#         valid_actions = env.valid_actions
#         distinct_actions = get_distinct_actions(env)
        
#         if self.is_full_after_my_turn(valid_actions, distinct_actions):
#             return (valid_actions[0], np.random.choice(range(self.action_size)))
        
#         if np.random.uniform() < self.eps:
#             return (np.random.choice(valid_actions), np.random.choice(valid_actions))
        
#         with torch.no_grad():
#             state_ = torch.FloatTensor(state).to(self.device)
#             # CNN일 때만 차원을 바꿔줌 
#             if self.policy_net.model_type == 'CNN':
#                 state_ = state_.reshape(6,7)
#                 state_ = state_.unsqueeze(0).unsqueeze(0)  # (6,7) -> (1,1,6,7)
#             else: state_ = state_.flatten()
            
            
            
#             # 여기선 q_value 가 49개임 
#             q_value = self.policy_net(state_).squeeze(0)
#             # print("state:",state)
#             # print("valid_actions:",valid_actions)
#             # print("q_value:",q_value)
#             a,b = self.get_minimax_action(q_value,valid_actions, distinct_actions)
            
#             return (a, b)
        
#     # replay buffer에 경험 추가 
#     def append_memory(self, state, a,b, reward, next_state, done):
#         if self.policy_net.model_type == 'Linear':
#             self.memory.append((state.flatten(), a, b, reward, next_state.flatten(), done))
#         else: 
#             self.memory.append((state.reshape(6,7), a, b, reward, next_state.reshape(6,7), done))
            


#     def train(self,epi,env,op_model):
#         env.reset()

#         is_op_myself = False
#         if op_model is None:
#             op_model = self
#             is_op_myself = True
#         # models 딕셔너리는 전역 변수로 사용하므로, players로 변경 
#         players = {1: self, 2: op_model}

#         for i in range(epi):
#             # 100번마다 loss, eps 등의 정보 표시
#             if i!=0 and i%100==0: 
#                 env.print_board(clear_board=False)
#                 print("epi:",i, ", agent's step:",self.steps)
#                 # 얼마나 학습이 진행되었는지 확인하기 위해, 모델 성능 측정 
                
#                 record = compare_model(self, op_model, n_battle=100)
#                 print(record)
#                 print("agent의 승률이 {}%".format(int(100*record[0]/sum(record))))
#                 print("loss:",sum(self.losses[-101:-1])/100.)
#                 print("epsilon:",self.eps)
#                 # simulate_model() 을 실행시키면 직접 행동 관찰 가능
#                 # simulate_model(self, op_model)

            
#             env.reset()

#             state_ = board_normalization(noise=self.noise_while_train, env=env, model_type=players[env.player].policy_net.model_type)
#             state = torch.from_numpy(state_).float()
#             done = False

#             past_state, past_action, past_reward, past_done = state, None, None, done
            
#             while not done:
#                 # 원래는 player, op_player 였지만, 직관적인 이해를 위해 수정 
#                 turn = env.player
#                 op_turn = 2//turn
                
#                 action = players[turn].select_action(state, env, player=turn)
                
#                 if isinstance(action, tuple):
#                     op_action_prediction = action[1]
#                     action = action[0]
                   
                
#                 observation, reward, done = env.step(action)
#                 op_state_ = board_normalization(noise=self.noise_while_train, env=env, model_type=players[turn].policy_net.model_type)
#                 op_state = torch.from_numpy(op_state_).float() 

#                 if past_action is not None:  # 맨 처음이 아닐 때 
#                     # 경기가 끝났을 때(중요한 경험)
#                     if done:
#                         repeat = 1
#                         # 중요한 경험일 때는 더 많이 memory에 추가해준다(optional)
#                         if reward > 0: repeat = self.repeat_reward
#                         for j in range(repeat):
#                             # 돌을 놓자마자 끝났으므로, next_state가 반전됨, 따라서 -1을 곱해준다
#                             if turn==1 or is_op_myself:
#                                 if action == None or op_action_prediction ==None:
#                                     print("this is impossible 3")
#                                     print(state,action,op_action_prediction)
#                                     exit()
#                                 self.append_memory(state,action,op_action_prediction, reward, op_state*-1, done)
#                                 # print for debugging
#                                 # print("for player")
#                                 # print("state:\n",torch.round(state).reshape(6,7).int())
#                                 # print("action:",action)
#                                 # print("reward:",reward)
#                                 # print("next_state\n",torch.round(op_state*-1).reshape(6,7).int())
#                                 # print()
#                             # 내가 이겼으므로 상대는 음의 보상을 받음 
#                             #Qmodels[op_player].append_memory(past_state, past_action, -reward, op_state, done)
#                             if turn==2 or is_op_myself:
#                                 if past_action == None or action ==None:
#                                     print("this is impossible 3")
#                                     print(state,past_action, action)
#                                     exit()
#                                 self.append_memory(past_state, past_action, action, -reward, op_state, done)
#                                 # print for debugging
#                                 # print("for opponent")
#                                 # print("state:\n",torch.round(past_state).reshape(6,7).int())
#                                 # print("action:",past_action)
#                                 # print("reward:",-reward)
#                                 # print("next_state\n",torch.round(op_state).reshape(6,7).int())
#                                 # print()


#                     # 경기가 끝나지 않았다면
#                     elif turn==2 or is_op_myself:  # 내 경험만 수집한다
#                         if past_action == None or action ==None:
#                                 print("this is impossible 3")
#                                 print(state,past_action, action)
#                                 exit()
#                         self.append_memory(past_state, past_action, action, past_reward, op_state, past_done)
#                         # print for debugging
#                         # print("for opponent")
#                         # print("state:\n",torch.round(past_state).reshape(6,7).int())
#                         # print("action:",past_action)
#                         # print("reward:",past_reward)
#                         # print("next_state\n",torch.round(op_state).reshape(6,7).int())
#                         # print()

                

#                 # info들 업데이트 해줌 
#                 past_state = state
#                 past_action = action
#                 past_reward = reward
#                 past_done = done
#                 state = op_state
                
#                 # replay buffer 를 이용하여 mini-batch 학습
#                 self.replay()
#                 # if Qagent.memory and abs(Qagent.memory[-1][2])!=1:
#                 #     print("state:\n",torch.round(Qagent.memory[-1][0]).int())
#                 #     print("action:",Qagent.memory[-1][1])
#                 #     print("reward:",Qagent.memory[-1][2])
#                 #     print("next_state\n",torch.round(Qagent.memory[-1][3]).int())
                
#                 # 게임이 끝났다면 나가기 
#                 if done: break
#             # print("eps:",Qagent.eps)
            
#             # epsilon-greedy
#             # min epsilon을 가지기 전까지 episode마다 조금씩 낮춰준다(1 -> 0.1)
#             if self.eps > 0.1: self.eps -= (1/epi)


#     def train_selfplay(self, epi, env, pool, add_pool):
#         env.reset()

#         # models 딕셔너리는 전역 변수로 사용하므로, players로 변경 
#         players = {1: self}

#         for i in range(epi):

#             players[2] = random.choice(pool)
#             # 100번마다 loss, eps 등의 정보 표시
#             if i!=0 and i%100==0: 
#                 env.print_board(clear_board=False)
#                 print("epi:",i, ", agent's step:",self.steps)
#                 # 얼마나 학습이 진행되었는지 확인하기 위해, 모델 성능 측정 
#                 record = compare_model(self, players[2], n_battle=100)
#                 print(record)
#                 print("agent의 승률이 {}%".format(int(100*record[0]/sum(record))))
#                 print("loss:",sum(self.losses[-101:-1])/100.)
#                 print("epsilon:",self.eps)
#                 # simulate_model() 을 실행시키면 직접 행동 관찰 가능
#                 # simulate_model(self, op_model)

            
#             env.reset()

#             state_ = board_normalization(noise=self.noise_while_train, env=env, model_type=players[env.player].policy_net.model_type)
#             state = torch.from_numpy(state_).float()
#             done = False

#             past_state, past_action, past_reward, past_done = state, None, None, done
            
#             while not done:
#                 # 원래는 player, op_player 였지만, 직관적인 이해를 위해 수정 
#                 turn = env.player
#                 op_turn = 2//turn
                
#                 action = players[turn].select_action(state, env, player=turn)

#                 if isinstance(action, tuple):
#                     op_action_prediction = action[1]
#                     action = action[0]

#                 observation, reward, done = env.step(action)
#                 op_state_ = board_normalization(noise=self.noise_while_train, env=env, model_type=players[turn].policy_net.model_type)
#                 op_state = torch.from_numpy(op_state_).float() 

#                 if past_action is not None:  # 맨 처음이 아닐 때 
#                     # 경기가 끝났을 때(중요한 경험)
#                     if done:
#                         repeat = 1
#                         # 중요한 경험일 때는 더 많이 memory에 추가해준다(optional)
#                         if reward > 0: repeat = self.repeat_reward
#                         for j in range(repeat):
#                             # 돌을 놓자마자 끝났으므로, next_state가 반전됨, 따라서 -1을 곱해준다
#                             if turn==1:
#                                 self.append_memory(state,action,op_action_prediction, reward, op_state*-1, done)
#                                 # print for debugging
#                                 # print("for player")
#                                 # print("state:\n",torch.round(state).reshape(6,7).int())
#                                 # print("action:",action)
#                                 # print("reward:",reward)
#                                 # print("next_state\n",torch.round(op_state*-1).reshape(6,7).int())
#                                 # print()
#                             # 내가 이겼으므로 상대는 음의 보상을 받음 
#                             #Qmodels[op_player].append_memory(past_state, past_action, -reward, op_state, done)
#                             if turn==2:
#                                 self.append_memory(past_state, past_action, action, -reward, op_state, done)
#                                 # print for debugging
#                                 # print("for opponent")
#                                 # print("state:\n",torch.round(past_state).reshape(6,7).int())
#                                 # print("action:",past_action)
#                                 # print("reward:",-reward)
#                                 # print("next_state\n",torch.round(op_state).reshape(6,7).int())
#                                 # print()


#                     # 경기가 끝나지 않았다면
#                     elif turn==2:  # 내 경험만 수집한다
#                         self.append_memory(past_state, past_action, action, past_reward, op_state, past_done)
#                         # print for debugging
#                         # print("for opponent")
#                         # print("state:\n",torch.round(past_state).reshape(6,7).int())
#                         # print("action:",past_action)
#                         # print("reward:",past_reward)
#                         # print("next_state\n",torch.round(op_state).reshape(6,7).int())
#                         # print()


#                 # info들 업데이트 해줌 
#                 past_state = state
#                 past_action = action
#                 past_reward = reward
#                 past_done = done
#                 state = op_state
                
#                 # replay buffer 를 이용하여 mini-batch 학습
#                 self.replay()
#                 if self.steps != 0 and self.steps%add_pool == 0:
#                     print("added in pool")
#                     new_model = copy.deepcopy(self)
#                     new_model.policy_net.eval()
#                     new_model.target_net.eval()
#                     new_model.eps = 0
#                     pool.append(new_model)
#                 # if Qagent.memory and abs(Qagent.memory[-1][2])!=1:
#                 #     print("state:\n",torch.round(Qagent.memory[-1][0]).int())
#                 #     print("action:",Qagent.memory[-1][1])
#                 #     print("reward:",Qagent.memory[-1][2])
#                 #     print("next_state\n",torch.round(Qagent.memory[-1][3]).int())
                
#                 # 게임이 끝났다면 나가기 
#                 if done: break
#             # print("eps:",Qagent.eps)
            
#             # epsilon-greedy
#             # min epsilon을 가지기 전까지 episode마다 조금씩 낮춰준다(1 -> 0.1)
#             if self.eps > 0.1: self.eps -= (1/epi)

#     # mini-batch로 업데이트 
#     def replay(self):
#         if len(self.memory) < self.batch_size*2:
#             return
        
#         random.shuffle(self.memory)
#         # batch size 만큼 랜덤으로 꺼낸다 
#         minibatch = random.sample(self.memory, self.batch_size)


#         if self.policy_net.model_type == 'Linear':
#             # state_batch.shape: (batch_size, 42)
#             state1_batch = torch.stack([s1 for (s1,a,b,r,s2,d) in minibatch]).to(self.device)
#             state2_batch = torch.stack([s2 for (s1,a,b,r,s2,d) in minibatch]).to(self.device)
#         elif self.policy_net.model_type == 'CNN':
#             # state_batch.shape: (batch_size, 1, 6, 7)
#             state1_batch = torch.stack([s1 for (s1,a,b,r,s2,d) in minibatch]).unsqueeze(1).to(self.device)
#             state2_batch = torch.stack([s2 for (s1,a,b,r,s2,d) in minibatch]).unsqueeze(1).to(self.device)

#         # action_batch.shape: (batch_size, )
#         action_batch = torch.Tensor([a for (s1,a,b,r,s2,d) in minibatch]).to(self.device)
#         op_action_batch = torch.Tensor([b for (s1,a,b,r,s2,d) in minibatch]).to(self.device)
#         action_index_batch = 7*action_batch + op_action_batch
#         reward_batch = torch.Tensor([r for (s1,a,b,r,s2,d) in minibatch]).to(self.device)
#         done_batch = torch.Tensor([d for (s1,a,b,r,s2,d) in minibatch]).to(self.device)
        
#         # print("state1_batch:",state1_batch.shape)
#         Q1 = self.policy_net(state1_batch)  # (256,7)
#         with torch.no_grad():
#             Q2 = self.target_net(state2_batch)

#         Q2 = Q2.reshape(-1, 7, 7)
#         # target Q value들 
#         Y = reward_batch + self.gamma * ((1-done_batch) * torch.max(torch.min(Q2, dim=2)[0], dim=1)[0])
#         # 해당하는 action을 취한 q value들
#         X = Q1.gather(dim=1,index=action_index_batch.long().unsqueeze(dim=1)).squeeze()
        
        
#         loss = nn.MSELoss()(X, Y.detach())
#         # print("to compare overestimation of Q value")
#         # print(state1_batch[200][0])
#         # print(state2_batch[200][0])
#         # print("action:",action_batch[200])
#         # print("reward:",reward_batch[200])
#         # print(Q1[200])
#         # print(Q2[200])
#         # print()

#         # tensor.numpy()는 cpu에서만 가능하므로 cpu() 처리
#         self.losses.append(loss.detach().cpu().numpy())
#         self.optimizer.zero_grad()
#         loss.backward()
#         self.optimizer.step()

#         # state, action, reward, next_state, done = zip(*minibatch)
#         # print(state)
#         # state = torch.FloatTensor(state) # .to(self.device)

#         # action = torch.LongTensor(action).unsqueeze(1)  #  .to(self.device)
#         # reward = torch.FloatTensor(reward).unsqueeze(1)  # .to(self.device)
#         # next_state = torch.FloatTensor(next_state)  # .to(self.device)
#         # done = torch.FloatTensor(done).unsqueeze(1)  # .to(self.device)
        
#         # current_q_values = self.policy_net(state).gather(1, action)
#         # next_q_values = self.target_net(next_state).detach().max(1)[0].unsqueeze(1)
#         # expected_q_values = reward + self.gamma * next_q_values * (1 - done)
        
#         # loss = nn.MSELoss()(current_q_values, expected_q_values)
#         # self.optimizer.zero_grad()
#         # loss.backward()
#         # self.optimizer.step()
#         self.steps += 1
#         if self.steps % self.target_update == 0:
#             print("update target net")
#             self.update_target_net()

        
#     # target net에 policy net 파라미터 들을 업데이트 해줌 
#     def update_target_net(self):
#         self.target_net.load_state_dict(self.policy_net.state_dict())


# 아래 heuristic agent는 (6,7) 보드판에서만 작동함
# model_num=3: Heuristic Model
# 랜덤으로 두다가 다음 step에 지거나 이길 수 있다면 행동함 
class HeuristicAgent():
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.policy_net = HeuristicModel()
        self.use_conv = True
        self.use_resnet = False
        self.use_minimax = False


    # normalize 된 board를 다시 1과 2로 바꿔줌
    def arr2board(self, state):
        state_ = copy.deepcopy(np.array(state.reshape(6,7)))
        state_ = np.round(state_).astype(int)
        state_[state_==-1] = 2

        return state_
    
    def put_piece(self, board, col, player):
        next_board = copy.deepcopy(board)
        for r in range(5, -1, -1):
            if next_board[r][col] == 0:
                next_board[r][col] = player
                break
        return next_board

    def select_action(self, state, env, player=1):
        valid_actions = env.valid_actions

        if state.ndim == 1: state = state.reshape(6,7)
        rows, cols = state.shape

        board = self.arr2board(state)
        # print(state, board)
        for col in valid_actions:

            new_board = self.put_piece(board, col, 1)

            # Check horizontally
            for r in range(rows):
                for c in range(cols - 3):
                    window = new_board[r, c:c+4]
                    if np.array_equal(window, [1, 1, 1, 1]):
                        return col

            # Check vertically
            for r in range(rows - 3):
                for c in range(cols):
                    window = new_board[r:r+4, c]
                    if np.array_equal(window, [1, 1, 1, 1]):
                        return col
                    
            # Check diagonally (down-right)
            for r in range(rows - 3):
                for c in range(cols - 3):
                    window = [new_board[r+i, c+i] for i in range(4)]
                    if np.array_equal(window, [1, 1, 1, 1]):
                        return col

            # Check diagonally (up-right)
            for r in range(3, rows):
                for c in range(cols - 3):
                    window = [new_board[r-i, c+i] for i in range(4)]
                    if np.array_equal(window, [1, 1, 1, 1]):
                        return col
                    
        for col in valid_actions:

            new_board = self.put_piece(board, col, 2)

            # Check horizontally
            for r in range(rows):
                for c in range(cols - 3):
                    window = new_board[r, c:c+4]
                    if np.array_equal(window, [2,2,2,2]):
                        return col

            # Check vertically
            for r in range(rows - 3):
                for c in range(cols):
                    window = new_board[r:r+4, c]
                    if np.array_equal(window, [2,2,2,2]):
                        return col
                    
            # Check diagonally (down-right)
            for r in range(rows - 3):
                for c in range(cols - 3):
                    window = [new_board[r+i, c+i] for i in range(4)]
                    if np.array_equal(window, [2,2,2,2]):
                        return col

            # Check diagonally (up-right)
            for r in range(3, rows):
                for c in range(cols - 3):
                    window = [new_board[r-i, c+i] for i in range(4)]
                    if np.array_equal(window, [2,2,2,2]):
                        return col

        return np.random.choice(valid_actions)



### m0nd2y
class ConnectFourRandomAgent(nn.Module) :
    def __init__(self, state_size=6*7, action_size=7, gamma=0.99, lr=0.003, batch_size=1024, target_update=1000, eps=1., memory_len=10000, model_num=4):
        super(ConnectFourRandomAgent,self).__init__()
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        ### Sangyeon
        # 모델의 type과 name을 지정해주기 위해 수정
        self.policy_net = RandomModel()
        
        # self.target_net = copy.deepcopy(self.policy_net)
        # self.target_net.load_state_dict(self.policy_net.state_dict())
        # self.target_net.eval()
        # self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = deque(maxlen=memory_len)
        # self.gamma = gamma  
        # self.state_size = state_size
        # self.action_size = action_size
        # self.target_update = target_update 
        # self.steps = 0
        self.eps = eps 
        self.batch_size = batch_size 
        # self.losses = [] 
        self.use_conv = False
        self.use_minimax = False
        self.use_resnet = False

    def select_action(self, state, env, player=1):
        return np.random.choice(env.valid_actions)
        
    def append_memory(self, state, action, reward, next_state, done):
        pass
        #self.memory.append((state, action, reward, next_state, done))
    
    def replay(self):
        if len(self.memory) < self.batch_size*2:
            return
        # minibatch = random.sample(self.memory, self.batch_size)

        # state1_batch = torch.stack([s1 for (s1,a,r,s2,d) in minibatch]).unsqueeze(1).to(self.device)
        # action_batch = torch.Tensor([a for (s1,a,r,s2,d) in minibatch]).to(self.device)
        # reward_batch = torch.Tensor([r for (s1,a,r,s2,d) in minibatch]).to(self.device)
        # state2_batch = torch.stack([s2 for (s1,a,r,s2,d) in minibatch]).unsqueeze(1).to(self.device)
        # done_batch = torch.Tensor([d for (s1,a,r,s2,d) in minibatch]).to(self.device)
        # Q1 = self.policy_net(state1_batch)
        # with torch.no_grad():
        #     Q2 = self.target_net(state2_batch)
        # Y = reward_batch + self.gamma * ((1-done_batch) * torch.max(Q2,dim=1)[0])
        # X = Q1.gather(dim=1,index=action_batch.long().unsqueeze(dim=1)).squeeze()
        # loss = nn.MSELoss()(X, Y.detach())
        # self.losses.append(loss.detach().cpu().numpy())
        # self.optimizer.zero_grad()
        # loss.backward()
        # self.optimizer.step()
        # self.steps += 1

    def update_target_net(self):
        pass
        #self.target_net.load_state_dict(self.policy_net.state_dict())




# https://github.com/smorga41/Connect-4-MiniMax-Ai/blob/main/Connect%204%20AI/Connect4_Ai.py
# evaluation을 위해 사용할 minimax agent, 위의 링크에서 조금 수정 
class MinimaxAgent():
    def __init__(self, depth=4):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.policy_net = MinimaxModel()
        self.use_conv = True
        self.use_resnet = False
        self.use_minimax = False

        self.depth = depth

        self.win_score = 10000000
        self.lose_score = -10000000

        self.four_score = 10000
        self.three_score = 10
        self.two_score = 3
        self.middle_column = 2
        self.aging_penalty = 3


        # 일단 op_four_score은 안쓸 예정 
        self.op_four_score = -10000000
        self.op_three_score = -12
        self.op_two_score = -4


    
    
    # 현재 보드 상태에서 이긴 플레이어가 있는지 확인
    def is_winner(self, board, player):
        rows, cols = board.shape
        # 수평 체크
        for row in range(rows):
            for col in range(cols - 3):
                if board[row][col] == player and board[row][col+1] == player and board[row][col+2] == player and board[row][col+3] == player:
                    return True

        # 수직 체크
        for row in range(rows - 3):
            for col in range(cols):
                if board[row][col] == player and board[row+1][col] == player and board[row+2][col] == player and board[row+3][col] == player:
                    return True

        # 오른쪽 위로 대각선 체크
        for row in range(rows - 3):
            for col in range(cols - 3):
                if board[row][col] == player and board[row+1][col+1] == player and board[row+2][col+2] == player and board[row+3][col+3] == player:
                    return True

        # 왼쪽 위로 대각선 체크
        for row in range(3, rows):
            for col in range(cols - 3):
                if board[row][col] == player and board[row-1][col+1] == player and board[row-2][col+2] == player and board[row-3][col+3] == player:
                    return True

        return False
    
   
    
    

    def score_position(self, board, player):
        score = 0
        rows, cols = board.shape
        ## Score center column
        center_array = [int(i) for i in list(board[cols//2,:])]
        center_count = center_array.count(player)
        score += center_count * 2

        ## Score Horizontal
        for r in range(rows):
            row_array = [int(i) for i in list(board[r,:])]
            for c in range(cols-3):
                window = row_array[c:c+4]
                score += self.evaluate_window(window, player)

        ## Score Vertical
        for c in range(cols):
            # print(board)
            # print(board[c,:])
            col_array = [int(i) for i in list(board[:,c])]
            for r in range(rows-3):
                window = col_array[r:r+4]
                score += self.evaluate_window(window, player)

        ## Score postive sloped diagonal
        for r in range(rows-3):
            for c in range(cols-3):
                window = [board[r+i][c+i] for i in range(4)]
                score += self.evaluate_window(window, player)

        for r in range(rows-3):
            for c in range(cols-3):
                window = [board[r+3-i][c+i] for i in range(4)]
                score += self.evaluate_window(window, player)

        return score


    def evaluate_window(self, window, player):
        score = 0
        op_player = 2//player

        if window.count(player) == 4:
            score += self.four_score
            return score
        elif window.count(player) == 3 and window.count(0) == 1:
            score += self.three_score
        elif window.count(player) == 2 and window.count(0) == 2:
            score += self.two_score

        if window.count(op_player) == 4:
            score += self.op_four_score
            return score
        elif window.count(op_player) == 3 and window.count(0) == 1:
            score += self.op_three_score
        elif window.count(op_player) == 2 and window.count(0) == 2:
            score += self.op_two_score
        return score

    def minimax(self, board, depth, alpha, beta, maximizingPlayer, player):
        valid_actions = get_valid_actions(board)

        
        if self.is_winner(board, 2):
            score = self.win_score + depth*self.aging_penalty
            return (None, score)
        elif self.is_winner(board, 1):
            score = self.lose_score - depth*self.aging_penalty
            return (None, score)
        elif not valid_actions: return (None, 0)
        elif depth == 0:
            return (None, self.score_position(board, player))


        if maximizingPlayer:
            value = -np.Inf
            action = random.choice(valid_actions)
            for col in valid_actions:
                new_board = np.copy(board)
                new_board = get_next_board(new_board, col, player)
                new_score = self.minimax(new_board, depth-1, alpha, beta, False, 2//player)[1]
                if new_score > value:
                    value = new_score
                    action = col

                alpha = max(alpha, value)
                if alpha >= beta:
                    break

            return (action, value)
        
        else:
            value = np.Inf
            action = random.choice(valid_actions)
            for col in valid_actions:
                new_board = np.copy(board)
                new_board = get_next_board(new_board, col, player)
                new_score = self.minimax(new_board, depth-1, alpha, beta, True, 2//player)[1]
                if new_score < value:
                    value = new_score
                    action = col

                beta = min(beta, value)
                if alpha >= beta:
                    break

            return (action, value)

    
    # 최적의 수(column) 반환
    def select_action(self, state, env, player):
        board = copy.deepcopy(env.board)
        depth = self.depth
        move, value = self.minimax(
            board,\
            depth, \
            maximizingPlayer=True, \
            alpha=-np.Inf, \
            beta=np.Inf, \
            player=player
        )
        # if value == 0:
        #     return np.random.choice(env.valid_actions)
        return move


class AlphaZero:
    def __init__(self, model, optimizer, game, args):
        self.model = model
        self.optimizer = optimizer
        self.game = game
        self.args = args
        self.mcts = MCTS(game, args, model)
        
    def selfPlay(self):
        memory = []
        player = 1
        state = self.game.get_initial_state()
        
        while True:
            neutral_state = self.game.change_perspective(state, player)
            action_probs = self.mcts.search(neutral_state)
            
            memory.append((neutral_state, action_probs, player))
            
            temperature_action_probs = action_probs ** (1 / self.args['temperature'])
            temperature_action_probs = temperature_action_probs / temperature_action_probs.sum()
            action = np.random.choice(self.game.action_size, p=temperature_action_probs) # change to temperature_action_probs
            
            state = self.game.get_next_state(state, action, player)
            
            value, is_terminal = self.game.get_value_and_terminated(state, action)
            
            if is_terminal:
                returnMemory = []
                for hist_neutral_state, hist_action_probs, hist_player in memory:
                    hist_outcome = value if hist_player == player else self.game.get_opponent_value(value)
                    returnMemory.append((
                        self.game.get_encoded_state(hist_neutral_state),
                        hist_action_probs,
                        hist_outcome
                    ))
                return returnMemory
            
            player = self.game.get_opponent(player)
                
    def train(self, memory):
        random.shuffle(memory)
        for batchIdx in range(0, len(memory), self.args['batch_size']):
            sample = memory[batchIdx:min(len(memory) - 1, batchIdx + self.args['batch_size'])] # Change to memory[batchIdx:batchIdx+self.args['batch_size']] in case of an error
            state, policy_targets, value_targets = zip(*sample)
            
            state, policy_targets, value_targets = np.array(state), np.array(policy_targets), np.array(value_targets).reshape(-1, 1)
            
            state = torch.tensor(state, dtype=torch.float32, device=self.model.device)
            policy_targets = torch.tensor(policy_targets, dtype=torch.float32, device=self.model.device)
            value_targets = torch.tensor(value_targets, dtype=torch.float32, device=self.model.device)
            
            out_policy, out_value = self.model(state)
            
            policy_loss = F.cross_entropy(out_policy, policy_targets)
            value_loss = F.mse_loss(out_value, value_targets)
            loss = policy_loss + value_loss
            
            self.optimizer.zero_grad() # change to self.optimizer
            loss.backward()
            self.optimizer.step() # change to self.optimizer
    
    def learn(self):
        for iteration in range(self.args['num_iterations']):
            memory = []
            
            self.model.eval()
            for selfPlay_iteration in range(self.args['num_selfPlay_iterations']):
                print("num_selfplay_iterations: {}/{}".format(\
                    selfPlay_iteration, self.args['num_selfPlay_iterations']
                ))
                memory += self.selfPlay()
                
            self.model.train()
            for epoch in range(self.args['num_epochs']):
                print("num_epochs: {}/{}".format(\
                    epoch, self.args['num_epochs']
                ))
                self.train(memory)
            
            torch.save(self.model.state_dict(), f"model_{iteration}_{self.game}.pt")
            torch.save(self.optimizer.state_dict(), f"optimizer_{iteration}_{self.game}.pt")

class AlphaZeroParallel:
    def __init__(self, model, optimizer, game, args):
        self.model = model
        self.optimizer = optimizer
        self.args = args
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR( \
            self.optimizer, 
            max_lr=0.2,
            steps_per_epoch=self.args['batch_size'], 
            epochs=self.args['num_epochs'],
            anneal_strategy='linear'
        )
        self.game = game
        
        self.mcts = MCTSParallel(game, args, model)
        self.steps = 0
        self.vlosses = []
        self.plosses = []
        self.losses = []
        
    def selfPlay(self):
        return_memory = []
        player = 1
        # num)parallel_games 만큼 게임을 만든다
        spGames = [SPG(self.game) for spg in range(self.args['num_parallel_games'])]
        
        while len(spGames) > 0:
            states = np.stack([spg.state for spg in spGames])
            neutral_states = self.game.change_perspective(states, player)
            
            # MCTS 형성 
            self.mcts.search(neutral_states, spGames)
            
            for i in range(len(spGames))[::-1]:
                spg = spGames[i]
                
                action_probs = np.zeros(self.game.action_size)

                # 방문 횟수를 비율로 해서 고른다. 이거도 ucb로 해서 고르면 안되나?
                for child in spg.root.children:
                    action_probs[child.action_taken] = child.visit_count
                    #action_probs[child.action_taken] = child.get_ucb(child)
                action_probs /= np.sum(action_probs)
                root_q = (spg.root.value_sum / spg.root.visit_count) 
                root_q = root_q[0] if isinstance(root_q, np.ndarray) else root_q
                # q와 z를 모두 target으로 이용하기 위해 memory 구조 변경 
                spg.memory.append((spg.root.state, action_probs, root_q, player))

                # temp가 클 수록 더 고른 분포를 고른다.
                temperature_action_probs = action_probs ** (1 / self.args['temperature'])
                temperature_action_probs /= np.sum(temperature_action_probs) # Divide temperature_action_probs with its sum in case of an error
                action = np.random.choice(self.game.action_size, p=temperature_action_probs) # Divide temperature_action_probs with its sum in case of an error

                spg.state = self.game.get_next_state(spg.state, action, player)

                value, is_terminal = self.game.get_value_and_terminated(spg.state, action)

                # 게임이 종료되면, memory에 state, policy, value 를 저장한다.
                # 일단 z와 q의 비율은 1대1로 두자 
                if is_terminal:
                    for hist_neutral_state, hist_action_probs, hist_value, hist_player in spg.memory:
                        hist_z = value if hist_player == player else self.game.get_opponent_value(value)
                        hist_outcome = (hist_z+hist_value)/2.
                        return_memory.append((
                            self.game.get_encoded_state(hist_neutral_state),
                            hist_action_probs,
                            hist_outcome
                        ))
                    del spGames[i]  # 메모리 관리를 위해 삭제 
                    
            player = self.game.get_opponent(player)
            
        return return_memory
                
    def train(self, memory):
        random.shuffle(memory)
        for batchIdx in range(0, len(memory), self.args['batch_size']):
            sample = memory[batchIdx:min(len(memory) - 1, batchIdx + self.args['batch_size'])] # Change to memory[batchIdx:batchIdx+self.args['batch_size']] in case of an error
            state, policy_targets, value_targets = zip(*sample)
            
            state, policy_targets, value_targets = np.array(state), np.array(policy_targets), np.array(value_targets).reshape(-1, 1)
            
            state = torch.tensor(state, dtype=torch.float32, device=self.model.device)
            policy_targets = torch.tensor(policy_targets, dtype=torch.float32, device=self.model.device)
            value_targets = torch.tensor(value_targets, dtype=torch.float32, device=self.model.device)
            
            out_policy, out_value = self.model(state)
            policy_loss = F.cross_entropy(out_policy, policy_targets)
            value_loss = F.mse_loss(out_value, value_targets)
            loss = 0.9*policy_loss + 0.1*value_loss
            
            self.plosses.append(policy_loss.item())
            self.vlosses.append(value_loss.item())
            self.losses.append(loss.item())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            self.steps += 1

            
    
    def learn(self):
        for iteration in range(self.args['num_iterations']):
            print("num_iterations: {}/{}".format(\
                    iteration+1, self.args['num_iterations']
                ))
            memory = []
            
            # selfplay 할때는 model을 evaluation 모드로 바꿔줌 
            self.model.eval()

            # selfplay 함수는 selfplay 도중에 저장한 메모리를 리턴한다 .
            for selfPlay_iteration in range(self.args['num_selfPlay_iterations'] // self.args['num_parallel_games']):
                print(" num_selfplay_iterations: {}/{}".format(\
                    selfPlay_iteration+1, self.args['num_selfPlay_iterations'] // self.args['num_parallel_games']
                ))
                memory += self.selfPlay()
            print(len(memory))
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR( \
                self.optimizer, 
                max_lr=0.2,
                steps_per_epoch=math.ceil(len(memory)/self.args['batch_size']), 
                epochs=self.args['num_epochs'],
                anneal_strategy='linear'
            )    
            # 저장한 memory들로 train할 때는 train 모드로 바꿔줌 
            self.model.train()

            for epoch in range(self.args['num_epochs']):
                print("    num_epochs: {}/{}".format(\
                    epoch+1, self.args['num_epochs']
                ))
                self.train(memory)
            
            self.save(iteration)
            
            # torch.save(self.model.state_dict(), f"model/alphazero/model_{iteration}_{self.game}.pth")
            # torch.save(self.optimizer.state_dict(), f"model/alphazero/optimizer_{iteration}_{self.game}.pth")
  
    def save(self,iter):
        # 저장할 폴더 찾고
        num = self.args['model_num']
        
        # args 저장하고
        self.args['train_time'] = get_current_time()
        with open('model/alphazero/model_{}/model_config_{}.json'.format(num,num), 'w') as f:
            json.dump(self.args, f, indent=4, ensure_ascii=False)
        
        # 모델 저장하고
        torch.save(self.model.state_dict(), "model/alphazero/model_{}/model_{}_iter_{}.pth".format(num,num,iter))

        
        # losses 저장하고
        plt.plot(self.losses)
        plt.savefig('model/alphazero/model_{}/loss_{}_iter{}.png'.format(num,num,iter))
        plt.close()
        plt.plot(self.vlosses)
        plt.savefig('model/alphazero/model_{}/vloss_{}_iter{}.png'.format(num,num,iter))
        plt.close()
        plt.plot(self.plosses)
        plt.savefig('model/alphazero/model_{}/ploss_{}_iter{}.png'.format(num,num,iter))
        plt.close()

# class AlphaZeroAgent:
#     def __init__(self, env, model_num=5, num_simulations=300, num_iterations=3, num_episodes=5, batch_size=16):
#         self.env = env
#         self.model = AlphaZeroResNet()
#         self.use_conv = True
#         self.batch_size = batch_size
#         self.num_simulations = num_simulations
#         self.num_iterations = num_iterations
#         self.num_episodes = num_episodes

#         self.mcts = MCTS(self.env, self.model, self.num_simulations)
#         self.memory = []
#         self.optimizer = optim.Adam(self.model.parameters(), lr=0.0003)
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         self.model.to(self.device)

#     def run_episode(self):
#         train_examples = []
#         player = 1
#         state = np.zeros((self.env.n_row, self.env.n_col))

#         while True:
#             # print(state)
#             perspective_state = self.env.get_perspective_state(state, player)

#             self.mcts = MCTS(self.env, self.model, self.num_simulations)
#             root = self.mcts.run(perspective_state, turn=1)

#             action_probs = np.zeros(self.env.action_size)
#             for key, value in root.children.items():
#                 action_probs[key] = value.visit_count

#             action_probs = action_probs / np.sum(action_probs)
#             train_examples.append((perspective_state, action_probs, player))

#             action = root.select_action(temp=0)
#             state, player = self.env.get_next_state(state, action, player)

#             reward = self.env.is_done(state, player)

#             # 게임이 끝났다면
#             if reward is not None:
#                 episode_record = []
#                 for state, action_probs, prev_player in train_examples:
#                     reward_record = reward * (-1)**(prev_player != player)
#                     episode_record.append((state, action_probs, reward_record))
            
#                 return episode_record
            

#     def train(self, epochs):
#         for iter in range(self.num_iterations):
#             print("iter:",iter)
#             for epi in range(self.num_episodes):
#                 print("epi:",epi)
#                 mem_epi = self.run_episode()
#                 self.memory.extend(mem_epi)

#             random.shuffle(self.memory)
            
#             prob_losses, value_losses = [], []

#             for epoch in range(epochs):
#                 print("epoch:",epoch)
#                 batch_num = 0

#                 while batch_num < int(len(self.memory) / self.batch_size):
#                     print("batch_num:", batch_num)
#                     sample_ids = np.random.randint(len(self.memory), size=self.batch_size)
#                     states, action_probs, values = list(zip(*[self.memory[i] for i in sample_ids]))

#                     states = torch.FloatTensor(np.array(states).astype(np.float64)).unsqueeze(1)
#                     target_probs = torch.FloatTensor(np.array(action_probs))
#                     target_values = torch.FloatTensor(np.array(values).astype(np.float64))

#                     states = states.contiguous().to(self.device)
#                     target_probs = target_probs.contiguous().to(self.device)
#                     target_values = target_values.contiguous().to(self.device)
#                     # print(states, states.shape)

#                     output_probs, output_values = self.model(states)
#                     loss_probs = self.get_loss_probs(target_probs, output_probs)
#                     loss_values = self.get_loss_values(target_values, output_values)
#                     loss = loss_probs + loss_values


#                     print("target_probs:", target_probs)
#                     print("output_probs:", output_probs)
#                     print("loss_probs: ", loss_probs)
#                     print("loss_values: ", loss_values)
#                     prob_losses.append(float(loss_probs))
#                     value_losses.append(float(loss_values))

#                     self.optimizer.zero_grad()
#                     loss.backward()
#                     self.optimizer.step()

#                     batch_num += 1

#                 print()
#                 print("policy loss: ", np.mean(prob_losses))
#                 print("value loss: ", np.mean(value_losses))

#     def get_loss_probs(self, target_probs, output_probs):
#         return -(target_probs * torch.log(output_probs + np.finfo(float).eps)).mean()
    
#     def get_loss_values(self, target_values, output_values):
#         return torch.sum((target_values-output_values.view(-1))**2)/target_values.size()[0]
    

