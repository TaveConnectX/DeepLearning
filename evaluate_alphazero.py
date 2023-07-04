
from functions import load_model, get_model_and_config_name, \
                    get_model_config
from agent_structure import ConnectFourRandomAgent, HeuristicAgent, MinimaxAgent
# from alphazero_new import ConnectFour, ResNet, MCTS
from env import ConnectFourEnv, board_normalization, ConnectFour, MCTS
from models import AlphaZeroResNet
import numpy as np
import torch
import random
import os 




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_search = True

CF = ConnectFour()
print("what the...")
args = {
    'C': 10,
    'num_searches': 343,
    'dirichlet_epsilon': 0.,
    'dirichlet_alpha': 0.3
}


player = np.random.choice([1,-1])


model = AlphaZeroResNet(5, 128).to(device)
model.load_state_dict(torch.load("model/alphazero/model_6/model_6_iter_2.pth", map_location=device))
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
                if not use_search:
                    encoded_state =  torch.tensor(get_encoded_state(state), device=device).unsqueeze(0)
                    action_probs, value = players[turn](encoded_state)
                    valid_moves = (state_[0] == 0).astype(np.uint8)
                    
                    action_probs = action_probs.detach().cpu().numpy() * valid_moves
                    action_probs /= np.sum(action_probs)
                    # print(state_)
                    # print(np.round(action_probs,3), value)
                    action = np.argmax(action_probs)
                
                else:
                    mcts_probs = mcts.search(state_)
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