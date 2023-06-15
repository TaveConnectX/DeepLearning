import numpy as np
import os
import random
import copy
import math
from collections import deque
import torch
from torch import nn, optim
import time
import env
from functions import compare_model, board_normalization, \
    get_model_config, set_optimizer, \
    get_distinct_actions, is_full_after_my_turn
from ReplayBuffer import RandomReplayBuffer
# models.py 분리 후 이동, 정상 작동하면 지울 듯 
# import torch.nn.init as init
# import torch.nn.functional as F
from models import DQNModel
import json
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


class ConnectFourDQNAgent(nn.Module):
    def __init__(self, env, state_size=6*7, action_size=7, config=None):
        super(ConnectFourDQNAgent,self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if config==None:
            config = get_model_config()

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
        self.eps = config['eps']  # DQN에서 사용될 epsilon
        self.batch_size = config['batch_size']  # size of mini-batch
        self.repeat_reward = config['repeat_reward']  # repeat reward
        self.losses = []  # loss들을 담는 list 


    def select_action(self, state, env, player=1):
        

        if self.use_minimax:
            distinct_actions = get_distinct_actions(env)
            
            if is_full_after_my_turn(env.valid_actions, distinct_actions):
                return (env.valid_actions[0], np.random.choice(range(self.action_size)))
            if np.random.uniform() < self.eps:
                return (np.random.choice(env.valid_actions), np.random.choice(env.valid_actions))
            
        
        else:
            if np.random.uniform() < self.eps:
                return np.random.choice(env.valid_actions)
            
        with torch.no_grad():
            state_ = torch.FloatTensor(state).to(self.device)
            # CNN일 때만 차원을 바꿔줌 
            if self.use_conv:
                state_ = state_.reshape(6,7)
                state_ = state_.unsqueeze(0).unsqueeze(0)  # (6,7) -> (1,1,6,7)
            else: state_ = state_.flatten()

            q_value = self.policy_net(state_)

            if self.use_minimax:
                # print("state:",state)
                # print("valid_actions:",valid_actions)
                # print("q_value:",q_value)
                a,b = self.get_minimax_action(q_value.squeeze(0),env.valid_actions, distinct_actions)
                return (a, b)
            
            else:
                # print("state:",state)
                # print("valid_actions:",valid_actions)
                # print("q_value:",q_value)
                valid_q_values = q_value.squeeze()[torch.tensor(env.valid_actions)]
                return env.valid_actions[torch.argmax(valid_q_values)]
        
    # # replay buffer에 경험 추가 
    # def append_memory(self, state, action, reward, next_state, done):
    #     if self.policy_net.model_type == 'Linear':
    #         self.memory.append((state.flatten(), action, reward, next_state.flatten(), done))
    #     else: 
    #         self.memory.append((state.reshape(6,7), action, reward, next_state.reshape(6,7), done))
            


    def train(self,epi,env:env.ConnectFourEnv,op_model):
        env.reset()


        if self.selfplay:
            players = {1: self}
            pool = [ConnectFourRandomAgent()]
        # models 딕셔너리는 전역 변수로 사용하므로, players로 변경 
        else: players = {1: self, 2: op_model}

        for i in range(epi):

            if self.selfplay: players[2] = random.choice(pool)
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

            state_ = board_normalization(noise=True, env=env, use_conv=players[env.player].use_conv)
            state = torch.from_numpy(state_).float()
            done = False

            past_state, past_action, past_reward, past_done = state, None, None, done
            
            while not done:
                # 원래는 player, op_player 였지만, 직관적인 이해를 위해 수정 
                turn = env.player
                op_turn = 2//turn
                
                action = players[turn].select_action(state, valid_actions=env.valid_actions, player=turn)
                
                if self.use_minimax:
                    op_action_prediction = action[1]
                    action = action[0]

                observation, reward, done = env.step(action)
                op_state_ = board_normalization(noise=True, env=env, use_conv=players[turn].use_conv)
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
                                    self.memory.add(state, action, op_action_prediction, reward, op_state*-1, done)
                                else:
                                    self.memory.add(state,action, reward, op_state*-1, done)
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
                                if self.use_minimax:
                                    self.memory.add(past_state, past_action, action, past_reward, op_state, past_done)
                                else:
                                    self.memory.add(past_state, past_action, -reward, op_state, done)
                                # print for debugging
                                # print("for opponent")
                                # print("state:\n",torch.round(past_state).reshape(6,7).int())
                                # print("action:",past_action)
                                # print("reward:",-reward)
                                # print("next_state\n",torch.round(op_state).reshape(6,7).int())
                                # print()


                    # 경기가 끝나지 않았다면
                    elif turn==2:  # 내 경험만 수집한다
                        if self.use_minimax:
                            self.memory.add(past_state, past_action, action, past_reward, op_state, past_done)
                        else:
                            self.memory.add(past_state, past_action, past_reward, op_state, past_done)
                        # print for debugging
                        # print("for opponent")
                        # print("state:\n",torch.round(past_state).reshape(6,7).int())
                        # print("action:",past_action)
                        # print("reward:",past_reward)
                        # print("next_state\n",torch.round(op_state).reshape(6,7).int())
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
                    new_model.eps = 0
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


    def train_selfplay(self, epi, env, pool, add_pool):
        env.reset()

        # models 딕셔너리는 전역 변수로 사용하므로, players로 변경 
        players = {1: self}

        for i in range(epi):

            players[2] = random.choice(pool)
            # 100번마다 loss, eps 등의 정보 표시
            if i!=0 and i%100==0: 
                env.print_board(clear_board=False)
                print("epi:",i, ", agent's step:",self.steps)
                # 얼마나 학습이 진행되었는지 확인하기 위해, 모델 성능 측정 
                record = compare_model(self, players[2], n_battle=100)
                print(record)
                print("agent의 승률이 {}%".format(int(100*record[0]/sum(record))))
                print("loss:",sum(self.losses[-101:-1])/100.)
                print("epsilon:",self.eps)
                # simulate_model() 을 실행시키면 직접 행동 관찰 가능
                # simulate_model(self, op_model)

            
            env.reset()

            state_ = board_normalization(noise=True, env=env, use_conv=players[env.player].use_conv)
            state = torch.from_numpy(state_).float()
            done = False

            past_state, past_action, past_reward, past_done = state, None, None, done
            
            while not done:
                # 원래는 player, op_player 였지만, 직관적인 이해를 위해 수정 
                turn = env.player
                op_turn = 2//turn
                
                action = players[turn].select_action(state, valid_actions=env.valid_actions, player=turn)
                if self.use_minimax:
                    op_action_prediction = action[1]
                    action = action[0]



                observation, reward, done = env.step(action)
                op_state_ = board_normalization(noise=True, env=env, use_conv=players[turn].use_conv)
                op_state = torch.from_numpy(op_state_).float() 

                if past_action is not None:  # 맨 처음이 아닐 때 
                    # 경기가 끝났을 때(중요한 경험)
                    if done:
                        repeat = 1
                        # 중요한 경험일 때는 더 많이 memory에 추가해준다(optional)
                        if reward > 0: repeat = self.repeat_reward
                        for j in range(repeat):
                            # 돌을 놓자마자 끝났으므로, next_state가 반전됨, 따라서 -1을 곱해준다
                            if turn==1:
                                if self.use_minimax:
                                    self.append_memory(state,action,op_action_prediction, reward, op_state*-1, done)
                                else:
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
                                if self.use_minimax:
                                    self.append_memory(past_state, past_action, action, -reward, op_state, done)
                                else:
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
                        if self.use_minimax:
                            self.append_memory(past_state, past_action, action, past_reward, op_state, past_done)
                        else:
                            self.append_memory(past_state, past_action, past_reward, op_state, past_done)
                        # print for debugging
                        # print("for opponent")
                        # print("state:\n",torch.round(past_state).reshape(6,7).int())
                        # print("action:",past_action)
                        # print("reward:",past_reward)
                        # print("next_state\n",torch.round(op_state).reshape(6,7).int())
                        # print()


                # info들 업데이트 해줌 
                past_state = state
                past_action = action
                past_reward = reward
                past_done = done
                state = op_state
                
                # replay buffer 를 이용하여 mini-batch 학습
                self.replay()
                if self.steps%add_pool == 0:
                    print("added in pool")
                    new_model = copy.deepcopy(self)
                    new_model.policy_net.eval()
                    new_model.target_net.eval()
                    new_model.eps = 0
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

    # mini-batch로 업데이트 
    def replay(self):
        if len(self.memory) < self.batch_size*2:
            return
        
        self.memory.shuffle()
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
        
        s_batch, a_batch, r_batch, s_prime_batch, d_batch = self.memory.sample()

        # print("state1_batch:",state1_batch.shape)
        Q1 = self.policy_net(s_batch)  # (256,7)
        with torch.no_grad():
            Q2 = self.target_net(s_prime_batch)

        # target Q value들 
        if self.use_minimax:
            Q2 = Q2.reshape(-1,7,7)
            Y = r_batch + self.gamma * ((1-d_batch) * torch.max(torch.min(Q2, dim=2)[0], dim=1)[0])
        else:
            Y = r_batch + self.gamma * ((1-d_batch) * torch.max(Q2,dim=1)[0])
        
        # 해당하는 action을 취한 q value들
        X = Q1.gather(dim=1,index=a_batch.long().unsqueeze(dim=1)).squeeze()
        
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



# DQN with minimax
class MinimaxDQNAgent(nn.Module):
    def __init__(self, state_size=6*7, action_size=7, gamma=0.99, lr=0.0001, batch_size=64, target_update=20, eps=1., memory_len=6400,repeat_reward=2,model_num=7):
        super(MinimaxDQNAgent,self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.policy_net = models[model_num]()
        self.target_net = copy.deepcopy(self.policy_net)
        # deepcopy하면 파라미터 load를 안해도 되는거 아닌가? 일단 두자
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()


        self.lr = lr
        # optimizer는 기본적으로 adam을 사용하겠지만 추후 다른 것으로 실험할 수도 있음
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        
        self.memory_len = memory_len
        # DQN에 사용될 replay memory(buffer)
        self.memory = deque(maxlen=self.memory_len)
        self.gamma = gamma  # discount factor
        self.state_size = state_size
        self.action_size = action_size
        self.target_update = target_update  # target net update 주기(여기선 step)
        self.steps = 0
        self.eps = eps  # DQN에서 사용될 epsilon
        self.batch_size = batch_size  # size of mini-batch
        self.repeat_reward = repeat_reward
        self.losses = []  # loss들을 담는 list 

    def get_distinct_actions(self, state, valid_actions):
        state_ = np.round(state)
        distinct_actions = []
        for a in valid_actions:
            if state_[1][a] != 0:
                distinct_actions.append(a)

        return distinct_actions


    # 한 칸만 남았으면 pair 액션이 불가능하므로 체크가 필요 
    def is_full_after_my_turn(self, valid_actions, distinct_actions):
        if len(valid_actions)==1 and len(distinct_actions)==1:
            return True
        else: return False

    def get_minimax_action(self, q_value,valid_actions, distinct_actions):
        q_dict = {}
        for a in valid_actions:
            q_dict[a] = (None, np.inf)
            for b in valid_actions:
                if a in distinct_actions and a==b: continue
                idx = 7*a + b
                # print(a,b)
                # print(q_value[idx])
                # print(q_dict[a][1])
                if q_value[idx] <= q_dict[a][1]:
                    q_dict[a] = (b, q_value[idx])

        max_key = None
        max_value = float('-inf')
        for a, (b, q) in q_dict.items():
            if q > max_value:
                max_key = a
                max_value = q

        return (max_key, q_dict[max_key][0])
    
    def select_action(self, state, valid_actions=None, player=1):
        if valid_actions is None:
            valid_actions = range(self.action_size)
        else:
            distinct_actions = self.get_distinct_actions(state, valid_actions)
        
        if self.is_full_after_my_turn(valid_actions, distinct_actions):
            return (valid_actions[0], np.random.choice(range(self.action_size)))
        
        if np.random.uniform() < self.eps:
            return (np.random.choice(valid_actions), np.random.choice(valid_actions))
        
        with torch.no_grad():
            state_ = torch.FloatTensor(state).to(self.device)
            # CNN일 때만 차원을 바꿔줌 
            if self.policy_net.model_type == 'CNN':
                state_ = state_.reshape(6,7)
                state_ = state_.unsqueeze(0).unsqueeze(0)  # (6,7) -> (1,1,6,7)
            else: state_ = state_.flatten()
            
            
            
            # 여기선 q_value 가 49개임 
            q_value = self.policy_net(state_).squeeze(0)
            # print("state:",state)
            # print("valid_actions:",valid_actions)
            # print("q_value:",q_value)
            a,b = self.get_minimax_action(q_value,valid_actions, distinct_actions)
            
            return (a, b)
        
    # replay buffer에 경험 추가 
    def append_memory(self, state, a,b, reward, next_state, done):
        if self.policy_net.model_type == 'Linear':
            self.memory.append((state.flatten(), a, b, reward, next_state.flatten(), done))
        else: 
            self.memory.append((state.reshape(6,7), a, b, reward, next_state.reshape(6,7), done))
            


    def train(self,epi,env,op_model):
        env.reset()

        is_op_myself = False
        if op_model is None:
            op_model = self
            is_op_myself = True
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
                
                if isinstance(action, tuple):
                    op_action_prediction = action[1]
                    action = action[0]
                   
                
                observation, reward, done = env.step(action)
                op_state_ = board_normalization(noise=True, env=env, model_type=players[turn].policy_net.model_type)
                op_state = torch.from_numpy(op_state_).float() 

                if past_action is not None:  # 맨 처음이 아닐 때 
                    # 경기가 끝났을 때(중요한 경험)
                    if done:
                        repeat = 1
                        # 중요한 경험일 때는 더 많이 memory에 추가해준다(optional)
                        if reward > 0: repeat = self.repeat_reward
                        for j in range(repeat):
                            # 돌을 놓자마자 끝났으므로, next_state가 반전됨, 따라서 -1을 곱해준다
                            if turn==1 or is_op_myself:
                                if action == None or op_action_prediction ==None:
                                    print("this is impossible 3")
                                    print(state,action,op_action_prediction)
                                    exit()
                                self.append_memory(state,action,op_action_prediction, reward, op_state*-1, done)
                                # print for debugging
                                # print("for player")
                                # print("state:\n",torch.round(state).reshape(6,7).int())
                                # print("action:",action)
                                # print("reward:",reward)
                                # print("next_state\n",torch.round(op_state*-1).reshape(6,7).int())
                                # print()
                            # 내가 이겼으므로 상대는 음의 보상을 받음 
                            #Qmodels[op_player].append_memory(past_state, past_action, -reward, op_state, done)
                            if turn==2 or is_op_myself:
                                if past_action == None or action ==None:
                                    print("this is impossible 3")
                                    print(state,past_action, action)
                                    exit()
                                self.append_memory(past_state, past_action, action, -reward, op_state, done)
                                # print for debugging
                                # print("for opponent")
                                # print("state:\n",torch.round(past_state).reshape(6,7).int())
                                # print("action:",past_action)
                                # print("reward:",-reward)
                                # print("next_state\n",torch.round(op_state).reshape(6,7).int())
                                # print()


                    # 경기가 끝나지 않았다면
                    elif turn==2 or is_op_myself:  # 내 경험만 수집한다
                        if past_action == None or action ==None:
                                print("this is impossible 3")
                                print(state,past_action, action)
                                exit()
                        self.append_memory(past_state, past_action, action, past_reward, op_state, past_done)
                        # print for debugging
                        # print("for opponent")
                        # print("state:\n",torch.round(past_state).reshape(6,7).int())
                        # print("action:",past_action)
                        # print("reward:",past_reward)
                        # print("next_state\n",torch.round(op_state).reshape(6,7).int())
                        # print()

                

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


    def train_selfplay(self, epi, env, pool, add_pool):
        env.reset()

        # models 딕셔너리는 전역 변수로 사용하므로, players로 변경 
        players = {1: self}

        for i in range(epi):

            players[2] = random.choice(pool)
            # 100번마다 loss, eps 등의 정보 표시
            if i!=0 and i%100==0: 
                env.print_board(clear_board=False)
                print("epi:",i, ", agent's step:",self.steps)
                # 얼마나 학습이 진행되었는지 확인하기 위해, 모델 성능 측정 
                record = compare_model(self, players[2], n_battle=100)
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

                if isinstance(action, tuple):
                    op_action_prediction = action[1]
                    action = action[0]

                observation, reward, done = env.step(action)
                op_state_ = board_normalization(noise=True, env=env, model_type=players[turn].policy_net.model_type)
                op_state = torch.from_numpy(op_state_).float() 

                if past_action is not None:  # 맨 처음이 아닐 때 
                    # 경기가 끝났을 때(중요한 경험)
                    if done:
                        repeat = 1
                        # 중요한 경험일 때는 더 많이 memory에 추가해준다(optional)
                        if reward > 0: repeat = self.repeat_reward
                        for j in range(repeat):
                            # 돌을 놓자마자 끝났으므로, next_state가 반전됨, 따라서 -1을 곱해준다
                            if turn==1:
                                self.append_memory(state,action,op_action_prediction, reward, op_state*-1, done)
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
                                self.append_memory(past_state, past_action, action, -reward, op_state, done)
                                # print for debugging
                                # print("for opponent")
                                # print("state:\n",torch.round(past_state).reshape(6,7).int())
                                # print("action:",past_action)
                                # print("reward:",-reward)
                                # print("next_state\n",torch.round(op_state).reshape(6,7).int())
                                # print()


                    # 경기가 끝나지 않았다면
                    elif turn==2:  # 내 경험만 수집한다
                        self.append_memory(past_state, past_action, action, past_reward, op_state, past_done)
                        # print for debugging
                        # print("for opponent")
                        # print("state:\n",torch.round(past_state).reshape(6,7).int())
                        # print("action:",past_action)
                        # print("reward:",past_reward)
                        # print("next_state\n",torch.round(op_state).reshape(6,7).int())
                        # print()


                # info들 업데이트 해줌 
                past_state = state
                past_action = action
                past_reward = reward
                past_done = done
                state = op_state
                
                # replay buffer 를 이용하여 mini-batch 학습
                self.replay()
                if self.steps != 0 and self.steps%add_pool == 0:
                    print("added in pool")
                    new_model = copy.deepcopy(self)
                    new_model.policy_net.eval()
                    new_model.target_net.eval()
                    new_model.eps = 0
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

    # mini-batch로 업데이트 
    def replay(self):
        if len(self.memory) < self.batch_size*2:
            return
        
        random.shuffle(self.memory)
        # batch size 만큼 랜덤으로 꺼낸다 
        minibatch = random.sample(self.memory, self.batch_size)


        if self.policy_net.model_type == 'Linear':
            # state_batch.shape: (batch_size, 42)
            state1_batch = torch.stack([s1 for (s1,a,b,r,s2,d) in minibatch]).to(self.device)
            state2_batch = torch.stack([s2 for (s1,a,b,r,s2,d) in minibatch]).to(self.device)
        elif self.policy_net.model_type == 'CNN':
            # state_batch.shape: (batch_size, 1, 6, 7)
            state1_batch = torch.stack([s1 for (s1,a,b,r,s2,d) in minibatch]).unsqueeze(1).to(self.device)
            state2_batch = torch.stack([s2 for (s1,a,b,r,s2,d) in minibatch]).unsqueeze(1).to(self.device)

        # action_batch.shape: (batch_size, )
        action_batch = torch.Tensor([a for (s1,a,b,r,s2,d) in minibatch]).to(self.device)
        op_action_batch = torch.Tensor([b for (s1,a,b,r,s2,d) in minibatch]).to(self.device)
        action_index_batch = 7*action_batch + op_action_batch
        reward_batch = torch.Tensor([r for (s1,a,b,r,s2,d) in minibatch]).to(self.device)
        done_batch = torch.Tensor([d for (s1,a,b,r,s2,d) in minibatch]).to(self.device)
        
        # print("state1_batch:",state1_batch.shape)
        Q1 = self.policy_net(state1_batch)  # (256,7)
        with torch.no_grad():
            Q2 = self.target_net(state2_batch)

        Q2 = Q2.reshape(-1, 7, 7)
        # target Q value들 
        Y = reward_batch + self.gamma * ((1-done_batch) * torch.max(torch.min(Q2, dim=2)[0], dim=1)[0])
        # 해당하는 action을 취한 q value들
        X = Q1.gather(dim=1,index=action_index_batch.long().unsqueeze(dim=1)).squeeze()
        
        
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


# 아래 heuristic agent는 (6,7) 보드판에서만 작동함
# model_num=3: Heuristic Model
# 랜덤으로 두다가 다음 step에 지거나 이길 수 있다면 행동함 
class HeuristicAgent():
    def __init__(self, model_num=3):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.policy_net = models[model_num]()


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

    def select_action(self, state, valid_actions=None, player=1):
        if valid_actions is None:
            valid_actions = range(self.action_size)
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
        self.policy_net = models[model_num]()
        
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

    def select_action(self, state, valid_actions=None, player=1):
        return np.random.choice(valid_actions)
        
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


            

class AlphaZeroAgent:
    def __init__(self, env:CFEnvforAlphaZero, model_num=5, num_simulations=300, num_iterations=3, num_episodes=5, batch_size=16):
        self.env = env
        self.model = models[model_num]()
        self.batch_size = batch_size
        self.num_simulations = num_simulations
        self.num_iterations = num_iterations
        self.num_episodes = num_episodes

        self.mcts = MCTS(self.env, self.model, self.num_simulations)
        self.memory = []
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0003)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def run_episode(self):
        train_examples = []
        player = 1
        state = np.zeros((self.env.row, self.env.col))

        while True:
            print(state)
            perspective_state = self.env.get_perspective_state(state, player)

            self.mcts = MCTS(self.env, self.model, self.num_simulations)
            root = self.mcts.run(perspective_state, turn=1)

            action_probs = np.zeros(self.env.action_size)
            for key, value in root.children.items():
                action_probs[key] = value.visit_count

            action_probs = action_probs / np.sum(action_probs)
            train_examples.append((perspective_state, action_probs, player))

            action = root.select_action(temp=0)
            state, player = self.env.get_next_state(state, action, player)

            reward = self.env.is_done(state, player)

            # 게임이 끝났다면
            if reward is not None:
                episode_record = []
                for state, action_probs, prev_player in train_examples:
                    reward_record = reward * (-1)**(prev_player != player)
                    episode_record.append((state, action_probs, reward_record))
            
                return episode_record
            

    def train(self, epochs):
        for iter in range(self.num_iterations):
            print("iter:",iter)
            for epi in range(self.num_episodes):
                print("epi:",epi)
                mem_epi = self.run_episode()
                self.memory.extend(mem_epi)

            random.shuffle(self.memory)
            
            prob_losses, value_losses = [], []

            for epoch in range(epochs):
                print("epoch:",epoch)
                batch_num = 0

                while batch_num < int(len(self.memory) / self.batch_size):
                    print("batch_num:", batch_num)
                    sample_ids = np.random.randint(len(self.memory), size=self.batch_size)
                    states, action_probs, values = list(zip(*[self.memory[i] for i in sample_ids]))

                    states = torch.FloatTensor(np.array(states).astype(np.float64)).unsqueeze(1)
                    target_probs = torch.FloatTensor(np.array(action_probs))
                    target_values = torch.FloatTensor(np.array(values).astype(np.float64))

                    states = states.contiguous().to(self.device)
                    target_probs = target_probs.contiguous().to(self.device)
                    target_values = target_values.contiguous().to(self.device)
                    # print(states, states.shape)

                    output_probs, output_values = self.model(states)
                    loss_probs = self.get_loss_probs(target_probs, output_probs)
                    loss_values = self.get_loss_values(target_values, output_values)
                    loss = loss_probs + loss_values


                    print("target_probs:", target_probs)
                    print("output_probs:", output_probs)
                    print("loss_probs: ", loss_probs)
                    print("loss_values: ", loss_values)
                    prob_losses.append(float(loss_probs))
                    value_losses.append(float(loss_values))

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    batch_num += 1

                print()
                print("policy loss: ", np.mean(prob_losses))
                print("value loss: ", np.mean(value_losses))

    def get_loss_probs(self, target_probs, output_probs):
        return -(target_probs * torch.log(output_probs + np.finfo(float).eps)).mean()
    
    def get_loss_values(self, target_values, output_values):
        return torch.sum((target_values-output_values.view(-1))**2)/target_values.size()[0]
    

