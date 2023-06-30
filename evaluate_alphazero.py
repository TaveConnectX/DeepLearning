
from functions import load_model, get_model_and_config_name, \
                    get_model_config
from agent_structure import ConnectFourRandomAgent, HeuristicAgent, MinimaxAgent
# from alphazero_new import ConnectFour, ResNet, MCTS
from env import ConnectFourEnv, board_normalization, ConnectFour, MCTS
from models import AlphaZeroResNet
import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def select_action(state, env, agent):
    print("output:\n")
    print("a prob:", agent.model(state)[0])
    print("value:", agent.model(state)[1])
    action_prob = agent.model(state)[0].detach().cpu().numpy()
    valid_actions = env.get_valid_actions(state[0][0].detach().cpu().numpy())
    action_prob *= valid_actions
    print(action_prob)
    action = np.argmax(action_prob)
    return action


def compare_model(model1, model2, n_battle=10):
    players = {1:model1, 2:model2}
    records = [0,0,0]  # model1 win, model2 win, draw
    comp_env = ConnectFourEnv()

    for round in range(n_battle):
        comp_env.reset()

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
                mcts_probs = mcts.search(state_)
                action = np.argmax(mcts_probs)
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


CF = ConnectFour()
print("what the...")
args = {
    'C': 2,
    'num_searches': 600,
    'dirichlet_epsilon': 0.,
    'dirichlet_alpha': 0.3
}


player = np.random.choice([1,-1])


model = AlphaZeroResNet(3, 64).to(device)
model.load_state_dict(torch.load("model/alphazero/model_1/model_1.pth", map_location=device))
model.eval()

mcts = MCTS(CF, args, model)

state = CF.get_initial_state()



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
    print(i, scores)
    evaluate_model(model, records, n_battles=n_battles)
    twos = records[0][-1]
    threes = records[1][-1]
    fours = records[2][-1]

    scores.append(twos*2+threes*3+fours*4)

print(records)
print(max(scores))
print(min(scores))
print(sum(scores)/len(scores))