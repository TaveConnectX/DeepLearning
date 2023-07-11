
from functions import load_model, get_model_and_config_name, \
                    get_model_config
from agent_structure import ConnectFourRandomAgent, HeuristicAgent, MinimaxAgent
# from alphazero_new import ConnectFour, ResNet, MCTS
from env import ConnectFourEnv, board_normalization, ConnectFour, MCTS, MCTS_alphago
from models import AlphaZeroResNet, ResNetforDQN
import numpy as np
import torch
import random
import os 
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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_search = False
alphazero = True
CF = ConnectFour()
print("what the...")
args = {
    'C': 1.5,
    'num_searches': 100,
    'dirichlet_epsilon': 0.,
    'dirichlet_alpha': 0.3
}


player = np.random.choice([1,-1])

# alphago 일 경우
if not alphazero:
    model = ResNetforDQN(num_blocks=5,num_hidden=128,action_size=49)
    model.load_state_dict(torch.load("model/model_81/Model81_DQN-resnet-minimax-v1.pth", map_location=device))
    model.eval()
    value_model = Classifier().to(device)
    value_model.load_state_dict(torch.load("model/models_for_V_net/ValueNetwork.pth", map_location=device))
    mcts = MCTS_alphago(CF, args, model,value_model=value_model)


# alphazero 일 경우
else:
    model_num, iter = 21, 4
    model = AlphaZeroResNet().to(device)
    model.load_state_dict(torch.load("model/alphazero/model_{}/model_{}_iter_{}.pth".format(model_num,model_num,iter), map_location=device))
    model.eval()
    mcts = MCTS(CF, args, model)

# state = CF.get_initial_state()

def seed_everything(seed: int = 42):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if device == "cuda:0":
        torch.cuda.manual_seed(seed)  # type: ignore
        torch.cuda.manual_seed_all(seed)
        # 이건 학습 속도가 줄어든다고 함 
        torch.backends.cudnn.deterministic = False  # type: ignore
        torch.backends.cudnn.benchmark = False  # type: ignore

seed_everything()


def get_encoded_state(state):
    encoded_state = np.stack(
        (state == -1, state == 0, state == 1)
    ).astype(np.float32)
    
    return encoded_state

def get_nash_prob_and_value(payoff_matrix, vas, iterations=50):
    if isinstance(payoff_matrix, torch.Tensor):    
        payoff_matrix = payoff_matrix.clone().detach().cpu().numpy().reshape(7,7)
    elif isinstance(payoff_matrix, np.ndarray):
        payoff_matrix = payoff_matrix.reshape(7,7)
    payoff_matrix = payoff_matrix[vas][:,vas]
    
    '''Return the oddments (mixed strategy ratios) for a given payoff matrix'''
    transpose_payoff = np.transpose(payoff_matrix)
    row_cum_payoff = np.zeros(len(payoff_matrix))
    col_cum_payoff = np.zeros(len(transpose_payoff))

    col_count = np.zeros(len(transpose_payoff))
    row_count = np.zeros(len(payoff_matrix))
    active = 0
    for i in range(iterations):
        row_count[active] += 1 
        col_cum_payoff += payoff_matrix[active]
        active = np.argmin(col_cum_payoff)
        col_count[active] += 1 
        row_cum_payoff += transpose_payoff[active]
        active = np.argmax(row_cum_payoff)
        
    value_of_game = (max(row_cum_payoff) + min(col_cum_payoff)) / 2.0 / iterations  
    row_prob = row_count / iterations
    col_prob = col_count / iterations
    
    return row_prob, col_prob, value_of_game

def compare_model(model1, model2, n_battle=10):
    players = {1:model1, 2:model2}
    records = [0,0,0]  # model1 win, model2 win, draw
    comp_env = ConnectFourEnv()
    
    for round in range(n_battle):
        comp_env.reset()
        a_cnt = 0
        while not comp_env.done:
            # 성능 평가이므로, noise를 주지 않음 
            turn = comp_env.player
            state_ = board_normalization(noise=False,env=comp_env, model_type='CNN')
            state = torch.from_numpy(state_).float()
            
            if turn == 2:
                
                
                action = players[turn].select_action(state, comp_env, player=turn)
                if isinstance(action, tuple):
                    action = action[0]
                
            else:
                a_cnt += 1
                # print("{},{}prev".format(round,a_cnt))
                valid_moves = (state_[0] == 0).astype(np.uint8)
                if not use_search:
                    encoded_state =  torch.tensor(get_encoded_state(state), device=device).unsqueeze(0)
                    if alphazero:
                        
                        action_probs, value = players[turn](encoded_state)
                        #print(action_probs, value, valid_moves)
                        action_probs = action_probs.detach().cpu().numpy() * valid_moves

                        action_probs /= np.sum(action_probs)
                        action = np.argmax(action_probs)
                        
                    else:                        
                        q_values = players[turn](encoded_state)
                        # print(q_values.reshape(7,7))
                        # prints()
                        vas = np.where(valid_moves==1)[0]
                        action_probs, value = mcts.get_minimax_prob_and_value(q_values, valid_moves)
                        # action_probs, op_action_probs, value = get_nash_prob_and_value(q_values,vas)
                        # action_probs = action_probs/action_probs.sum()
                        
                        # action = np.random.choice(vas, p=action_probs)
                        action = vas[np.argmax(action_probs)]
                        # print(state)
                        # print(action_probs, op_action_probs, value)
                        # print(action)
                        
                    
                    
                    
                    # print(state_)
                    # print(np.round(action_probs,3), value)
                    
                
                else:
                    mcts_probs = mcts.search(np.array(state_))
                    action = np.argmax(mcts_probs)
                # print("{}after".format(a_cnt))
                # print(mcts_probs, action)

            comp_env.step(action)
        
        if comp_env.win == 1: records[0] += 1
        elif comp_env.win == 2: records[1] += 1
        else: records[2] += 1

    return records


def evaluate_model(model, record, n_battles=[10,10,10]):
    op_agents = [ConnectFourRandomAgent(), HeuristicAgent(), MinimaxAgent()]
    
    
    w,l,d = compare_model(model, op_agents[0], n_battle=n_battles[0])
    record[0].append(w+d)
    w,l,d = compare_model(model, op_agents[1], n_battle=n_battles[1])
    record[1].append(w+d)
    w,l,d = compare_model(model, op_agents[2], n_battle=n_battles[2])
    record[2].append(w+d)




# 수능처럼 점수를 계산하자
# random model: 2점
# heuristic model: 3점
# minimax model: 4점

# 2점 3
# 3점 14
# 4점 13

# compare model로 점수 계산 
num_tests = 10
n_battles = [3,14,13]
records = [[],[],[]]
scores = []
for i in range(num_tests):
    
    
    evaluate_model(model, records, n_battles=n_battles)
    twos = records[0][-1]
    threes = records[1][-1]
    fours = records[2][-1]
    print(records[0][-1], records[1][-1], records[2][-1])

    scores.append(twos*2+threes*3+fours*4)
    print(i, scores)

print(records)
print(max(scores))
print(min(scores))
print("avg:",sum(scores)/len(scores))