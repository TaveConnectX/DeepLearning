import torch
import numpy as np
import math
from env import ConnectFour
from models import ResNetforDQN
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
            # child와 parent는 적이므로 1에서 빼주기로 한다 
            q_value = 1 - ((child.value_sum / child.visit_count) + 1) / 2
        return q_value + self.args['C'] * (math.sqrt(self.visit_count) / (child.visit_count + 1)) * child.prior
    
    def expand(self, policy):
        for action, prob in enumerate(policy):
            if prob > 0:
                child_state = self.state.copy()
                # 내가 두는 건 항상 1, child 는 -1. 재귀적으로 계속 바뀜 
                # print(child_state, action,policy, action, prob)
                child_state = self.game.get_next_state(child_state, action, 1)
                child_state = self.game.change_perspective(child_state, player=-1)
                # game, args, state, parent=None, action_taken=None, prior=0, visit_count=0
                child = Node(
                    game=self.game, 
                    args=self.args, 
                    state=child_state, 
                    parent=self, 
                    action_taken=action, 
                    prior=prob
                )
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
        
    def board_normalization(self,state):
        return torch.tensor(state, device=self.model.device).float().unsqueeze(0).unsqueeze(0)

    def get_nash_prob_and_value(self,payoff_matrix, vas, iterations=10):
        if isinstance(payoff_matrix, torch.Tensor):    
            payoff_matrix = payoff_matrix.clone().detach().reshape(7,7)
        elif isinstance(payoff_matrix, np.ndarray):
            payoff_matrix = payoff_matrix.reshape(7,7)
        vas = np.where(np.array(vas) == 1)[0]
        payoff_matrix = payoff_matrix[vas][:,vas]
        # print("vas:",vas)
        '''Return the oddments (mixed strategy ratios) for a given payoff matrix'''
        transpose_payoff = torch.transpose(payoff_matrix,0,1)
        row_cum_payoff = torch.zeros(len(payoff_matrix)).to(self.model.device)
        col_cum_payoff = torch.zeros(len(transpose_payoff)).to(self.model.device)

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
    

    @torch.no_grad()
    def search(self, state):
        root = Node(self.game, self.args, state, visit_count=1)
        
        q_values = self.model(
            self.board_normalization(state)
        )
        valid_moves = self.game.get_valid_moves(state)
        # print(q_values)
        # print(valid_moves)
        pa, pb, v = self.get_nash_prob_and_value(q_values, valid_moves)
        policy = np.zeros_like(valid_moves, dtype=float)
        policy[np.array(valid_moves) == 1] = pa
        print(policy, v)
        # print(np.array(valid_moves) == 1,policy,pa,pb, v)
        # policy = torch.softmax(policy, axis=1).squeeze(0).cpu().numpy()
        policy = (1 - self.args['dirichlet_epsilon']) * policy + self.args['dirichlet_epsilon'] \
            * np.random.dirichlet([self.args['dirichlet_alpha']] * self.game.action_size)
        
        
        
        # policy *= valid_moves
        # policy /= np.sum(policy)
        root.expand(policy)
        
        for search in range(self.args['num_searches']):
            node = root
            
            while node.is_fully_expanded():
                node = node.select()
                
            value, is_terminal = self.game.get_value_and_terminated(node.state, node.action_taken)
            value = self.game.get_opponent_value(value)
            
            if not is_terminal:
                q_values = self.model(
                    self.board_normalization(node.state)
                )
                valid_moves = self.game.get_valid_moves(node.state)
                # print(node.state, valid_moves)
                # print(q_values)
                # print(valid_moves)
                pa, pb, value = self.get_nash_prob_and_value(q_values, valid_moves)
                policy = np.zeros_like(valid_moves, dtype=float)
                policy[np.array(valid_moves) == 1] = pa
                # print(policy,pb, value)
                # policy = torch.softmax(policy, axis=1).squeeze(0).cpu().numpy()
                # valid_moves = self.game.get_valid_moves(node.state)
                # policy *= valid_moves
                # policy /= np.sum(policy)
                
                value = value.item()
                node.expand(policy)
                
            node.backpropagate(value)    
            
            
        action_probs = np.zeros(self.game.action_size)
        # action prob은 방문 횟수에 비례하도록 정한다 
        for child in root.children:
            action_probs[child.action_taken] = child.visit_count
        action_probs /= np.sum(action_probs)
        return action_probs
    
CF = ConnectFour()
args = {
    'C': 4,
    'num_searches': 800,
    'dirichlet_epsilon': 0.,
    'dirichlet_alpha': 0.3,
    'temperature':2
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNetforDQN(action_size=49)
model.load_state_dict(torch.load("model/model_9/selfplayModel9_DQN-ResNet-v1.pth", map_location=device))
model.eval()

mcts = MCTS(CF, args, model)

state = CF.get_initial_state()
# state = torch.from_numpy(state).float().unsqueeze(0).unsqueeze(0)
# state = state.to(device)
# print(model(state))

# action_probs = mcts.search(state)
# print(action_probs)
player = np.random.choice([1,-1])

while True:
    print(state)
    if player == 1:
        # print("state", state)
        # print("state", state[0][0].detach().cpu().numpy())
        valid_moves = CF.get_valid_moves(state)
        print("valid_moves", [i for i in range(CF.action_size) if valid_moves[i] == 1])
        action = int(input(f"{player}:"))

        if valid_moves[action] == 0:
            print("action not valid")
            continue

            
    else:
        neutral_state = CF.change_perspective(state, player)
        mcts_probs = mcts.search(neutral_state)
        print("player -1:",mcts_probs)
        # time.sleep(5)
        action = np.argmax(mcts_probs)
        # action = np.random.choice(range(7),p=mcts_probs)
        
    state = CF.get_next_state(state, action, player)
    
    value, is_terminal = CF.get_value_and_terminated(state, action)
    
    if is_terminal:
        print(state)
        if value == 1:
            print(player, "won")
        else:
            print("draw")
        break
        
    player = CF.get_opponent(player)