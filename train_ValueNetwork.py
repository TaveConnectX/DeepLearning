import env
import AlphaZeroenv
import copy
import json
import numpy as np
import torch
import random
from collections import deque
from torch import nn, optim
import time
import matplotlib.pyplot as plt
import os
from models import DQNModel
from ReplayBuffer import RandomReplayBuffer
from functions import get_model_config, save_model, \
                        get_current_time, get_model_and_config_name, load_model, board_normalization, \
                        set_optimizer
from agent_structure import ConnectFourDQNAgent, HeuristicAgent, set_op_agent, evaluate_model
import random

    
#Value_Network
class ConnectFourValueAgent(nn.Module) :
    def __init__(self, state_size=6*7, action_size=7, config_file_name=None, **kwargs):
        super(ConnectFourValueAgent,self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        config = get_model_config(config_file_name)
        for key, value in kwargs.items():
            config[key] = value

        self.use_conv=config['use_conv']
        self.use_resnet=config['use_resnet']
        self.use_minimax=config['use_minimax']

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
        self.softmax_const = config['softmax_const']
        self.max_temp = config['max_temp']
        self.min_temp = config['min_temp']
        self.temp_decay = (self.min_temp/self.max_temp)**(1/(self.epi*self.softmax_const))
        self.temp = self.max_temp
        self.eps = config['eps']  # DQN에서 사용될 epsilon
        
        self.batch_size = config['batch_size']  # size of mini-batch
        self.repeat_reward = config['repeat_reward']  # repeat reward
        self.losses = []  # loss들을 담는 list 
        # record of win rate of random, heuristic, minimax agent
        self.record = [[], [], []]

    def train(self,epi) :
        # 1. Supervised Learning Model 불러오기
        # models_for_V_net 에서 모델과 config 를 불러온다.
        folder_path = 'model/models_for_V_net'
        SL_model_name, SL_model_config = get_model_and_config_name(folder_path+'/model_SL')
        # print(SL_model_name,SL_model_config)
        # 불러온 config 파일로 모델 껍데기를 만듦
        SL_agent = ConnectFourDQNAgent(config_file_name=folder_path+'/model_SL/'+SL_model_config)
        # 불러온 모델 파일로 모델 업로드
        load_model(SL_agent.policy_net, filename=folder_path+'/model_SL/'+SL_model_name)

        # 2. Reinforcement Learning Model 불러오기
        RL_model_name, RL_model_config = get_model_and_config_name(folder_path+'/model_RL')
        # print(RL_model_name,RL_model_config)
        RL_agent = ConnectFourDQNAgent(config_file_name=folder_path+'/model_RL/'+RL_model_config)
        load_model(RL_agent.policy_net, filename=folder_path+'/model_RL/'+RL_model_name)

        SL_agent.eps, RL_agent.eps = 0.0, 0.0  # greedy 한 action을 취하기 위해서

        # 환경 생성 
        VEnv = env.ConnectFourEnv()

        VEnv.reset()

        # models 딕셔너리는 전역 변수로 사용하므로, players로 변경 
        #players = {1: self, 2: op_model}

        for i in range(epi):
            state_ = board_normalization(noise=False, env=VEnv, use_conv=SL_agent.use_conv)
            state = torch.from_numpy(state_).float()

            # 100번마다 loss, eps 등의 정보 표시
            if i!=0 and i%100==0: 
                #env.print_board(clear_board=False)
                print("epi:",i, ", agent's step:",self.steps)
                # 얼마나 학습이 진행되었는지 확인하기 위해, 모델 성능 측정 
                #evaluate_model(self, record=self.record)
                
                #print(self.record[0][-1],self.record[1][-1], self.record[2][-1])
                # print("agent의 승률이 {}%".format(int(100*record[0]/sum(record))))
                print("loss:",sum(self.losses[-101:-1])/100.)
                if self.eps > self.softmax_const: print("epsilon:",self.eps)
                #else: print("temp:",self.temp)
                # simulate_model() 을 실행시키면 직접 행동 관찰 가능
                # simulate_model(self, op_model)

            
            VEnv.reset()

            state_ = board_normalization(noise=False, env=VEnv, use_conv=SL_agent.use_conv)
            state = torch.from_numpy(state_).float()
            use_rl = False  # RL 에이전트를 사용할 턴인지 여부
            rl_start_turn = 19  # RL 에이전트를 사용하기 시작하는 턴
            
            for turn in range(1, 41):
                if turn < 19:
                    action = SL_agent.select_action(state=state, env=VEnv, player=VEnv.player)
                    if SL_agent.use_minimax: 
                        op_action_prediction = action[1]  # 상대 예상 액션은 이 파일에서 필요 없을 것으로 보임(아마?)
                        action = action[0]  # action은 이제 0~6 의 정수가 됨 
                    import_state = state
                # RL 에이전트 사용 (19 수 이후)
                else:
                    if use_rl:
                        action = RL_agent.select_action(state=state, env=VEnv, player=VEnv.player)
                        if RL_agent.use_minimax: 
                            op_action_prediction = action[1]  # 상대 예상 액션은 이 파일에서 필요 없을 것으로 보임(아마?)
                            action = action[0]  # action은 이제 0~6 의 정수가 됨 
                    else:
                        action = random.randint(0, 6)  # 랜덤으로 액션 선택

                #print("Selected Action:", action)

                # 선택한 action으로 환경 진행
                state, reward, done = VEnv.step(action)

                #print("Observation:", state)
                #print("Reward:", reward)
                #print("Done:", done)
                #print("Current Player:", VEnv.player)

                # RL 에이전트 사용 여부 결정
                if turn == rl_start_turn:
                    import_state = state
                    use_rl = True

                if done:
                        # 돌을 놓자마자 끝났으므로, next_state가 반전됨, 따라서 -1을 곱해준다
                    if turn%2==1:
                        if self.use_minimax:
                            self.memory.add(import_state, action, op_action_prediction, reward, state*-1, done)
                        else:
                            self.memory.add(import_state, action, reward, state*-1, done)
                    if turn%2==0:
                        if self.use_minimax:
                            self.memory.add(import_state, action, op_action_prediction, -reward, state*-1, done)
                        else:
                            self.memory.add(import_state, action, -reward, state*-1, done)                
                # 게임이 끝났다면 나가기 
                if done:
                    break
            # print("eps:",Qagent.eps)
            
            # epsilon-greedy
            # min epsilon을 가지기 전까지 episode마다 조금씩 낮춰준다(1 -> 0.1)
            if self.eps > 0.1: self.eps -= (1/epi)
            if self.eps < self.softmax_const:
                self.temp *= self.temp_decay


if __name__ == "__main__" :
    config = get_model_config()
    CFenv = env.ConnectFourEnv()  # connext4 환경 생성
    Qagent = ConnectFourValueAgent(
        state_size=CFenv.n_col * CFenv.n_row,
        action_size=CFenv.n_col
    )
    Qagent.train(epi=config['epi'])