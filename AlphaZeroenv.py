
import torch
import math
import random
import numpy as np

# env.py 의 ConnectFourEnv와는 다르게 게임 규칙만 들어가 있음 
# player는 1과 -1로 이루어져 있으므로 정규화할 필요 없음 
# self play 이므로 초기 player를 랜덤으로 둘 필요 없음 
# action은 one-hot encoding 된 상태로 받음 ex. [0,0,1,0,0,0,0]
class CFEnvforAlphaZero:
    def __init__(self, row=6, col=7, action_size=7):
        self.n_row = row
        self.n_col = col
        self.board = np.zeros((row, col))
        self.action_size = action_size

    def get_next_state(self, state, action, player):
        board = np.copy(state)
        col = int(action)
        if not board[0,col] == 0:
            print(board, col, player)
            print("1:this cannot be happened")
        else:
            # piece를 둠 
            for row in range(self.n_row-1,-1,-1):
                if board[row][col] == 0:
                    board[row][col] = player
                    break
                else: continue
        
        return board, player * -1
    
    def get_valid_actions(self, state):
        return np.where(state[0] == 0, 1, 0)
    
    def is_done(self, state, player):
        for i in range(self.n_row):
            for j in range(self.n_col):
                if state[i][j] == player:
                    # horizontal
                    if j + 3 < self.n_col and state[i][j+1] == state[i][j+2] == state[i][j+3] == player:
                        return player
                    # vertical
                    if i + 3 < self.n_row and state[i+1][j] == state[i+2][j] == state[i+3][j] == player:
                        return player
                    # diagonal (down right)
                    if i + 3 < self.n_row and j + 3 < self.n_col and state[i+1][j+1] == state[i+2][j+2] == state[i+3][j+3] == player:
                        return player
                    # diagonal (up right)
                    if i - 3 >= 0 and j + 3 < self.n_col and state[i-1][j+1] == state[i-2][j+2] == state[i-3][j+3] == player:
                        return player
    
        
        # 맨 윗줄이 모두 꽉차있다면, 비긴 것
        if not 0 in state[0,:]:
            return 0
        
        # game이 아직 끝나지 않았을 때 
        return None

    def get_perspective_state(self, state, player):
        return state * player

# def get_next_state(state, action, player):
#         board = np.copy(state)
#         col = int(action)
#         len_row = len(board)
#         if not board[0,col] == 0:
#             print(board, col, player)
#             print("1:this cannot be happened")
#         else:
#             # piece를 둠 
#             for row in range(len_row-1,-1,-1):
#                 if board[row][col] == 0:
#                     board[row][col] = player
#                     break
#                 else: continue
        
#         return board, player * -1


# def is_done(state):
#         row = len(state)
#         col = len(state[0])
#         for player in [1,-1]:
#             for i in range(row):
#                 for j in range(col):
#                     if state[i][j] == player:
#                         # horizontal
#                         if j + 3 < col and state[i][j+1] == state[i][j+2] == state[i][j+3] == player:
#                             return player
#                         # vertical
#                         if i + 3 < row and state[i+1][j] == state[i+2][j] == state[i+3][j] == player:
#                             return player
#                         # diagonal (down right)
#                         if i + 3 < row and j + 3 < col and state[i+1][j+1] == state[i+2][j+2] == state[i+3][j+3] == player:
#                             return player
#                         # diagonal (up right)
#                         if i - 3 >= 0 and j + 3 < col and state[i-1][j+1] == state[i-2][j+2] == state[i-3][j+3] == player:
#                             return player

#         # 맨 윗줄이 모두 꽉차있다면, 비긴 것
#         if not 0 in state[0,:]:
#             return 0
#         # game이 아직 끝나지 않았을 때 
#         return None

# class Node:
#     def __init__(self, prior, turn, state):
#         self.prior = prior
#         self.turn = turn
#         self.state = state
#         self.children = {}
#         self.visits = 0
#         self.value = 0

#     def expand(self, action_probs):
#         for action, prob in enumerate(action_probs):
#             if prob==0: continue
#             next_state, next_turn = get_next_state(self.state, action, self.turn)
#             self.children[action] = Node(prob, next_turn, next_state)

#     def select_child(self):
#         max_score = float('-inf')
#         for action, child in self.children.items():
#             score = ucb_score(self, child)
#             if score > max_score:
#                 selected_action = action
#                 selected_child = child
#                 max_score = score

#         return selected_action, selected_child
    



# def dummy_model_predict(state):
#     return 0.5, [0.5,0,0,0,0,0.5,0]
# def ucb_score(parent, child):
#     prior_score = child.prior * math.sqrt(parent.visits) / (child.visits+1)

#     if child.visits == 0: value_score = 0
#     else: value_score = child.value / child.visits

#     return value_score + prior_score


# board = np.array([[0,1,1,1,-1,0,1],
#                   [0,-1,1,-1,-1,0,-1],
#                   [1,-1,1,-1,-1,0,1],
#                   [-1,1,-1,1,1,0,1],
#                   [1,1,-1,1,-1,-1,1],
#                   [1,-1,-1,1,-1,1,-1]])
# board *= -1
# root = Node(prior=0, turn=1, state=board)
# value, action_probs =dummy_model_predict(board)

# root.expand(action_probs=action_probs)


# for _ in range(100):
#     node = root
#     search_path = [node]

#     while len(node.children) > 0:
#         action, node = node.select_child()
#         search_path.append(node)

#     value = None
#     done = is_done(node.state)
#     if done==0:
#         value = 0
#     elif done==1:
#         value = 1
#     elif done==-1:
#         value = -1

#     if value is None:
#         value, action_probs = dummy_model_predict(node.state)
#         node.expand(action_probs)

#     for node in search_path:
#         node.value += value
#         node.visits += 1



class Node:
    def __init__(self, prior_prob, turn, state=None):
        self.state = state
        self.turn = turn

        self.visit_count = 0
        self.value_sum = 0
        self.prior_prob = prior_prob

        
        # (action, child) 가 담겨있는 dict
        self.children = {}

    def get_value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    def select_action(self, temp):
        # 현 node 에서 어떤 action을 취했을때 발생하는 child에 대한 visit_count 모음음
        visit_counts = np.array([
            child.visit_count for child in self.children.values()
            ])
        
        actions = np.array([
            action for action in self.children.keys()
            ])

        if temp == 0: action = actions[np.argmax(visit_counts)]
        elif temp == np.inf: action = np.random.choice(actions)
        else:
            # visit count distribution을 temperature을 이용해서 계산 
            vc_dist = visit_counts ** (1 / temp)
            # 다시 다 더하면 1로 정규화(확률이므로)
            vc_dist = vc_dist / vc_dist.sum()
            # 새로 만든 distribution을 이용해서 action sampling
            action = np.random.choice(actions, p=vc_dist)

        return action

    def select_child(self):
        best_ucb = -np.inf
        best_action = -1  # where can I use best_action?
        best_child = None

        for action, child in self.children.items():
            ucb = self.get_ucb(child)
            if ucb > best_ucb:
                best_ucb = ucb
                best_action = action
                best_child = child

        return best_action, best_child
    
    def get_ucb(self, child, c_puct=2):
        prior = child.prior_prob * math.sqrt(self.visit_count) / (child.visit_count + 1)
        # child 와 node는 적대적 관계이므로 음수를 곱해줌
        value = - child.get_value()

        
        ucb = value + c_puct * prior

        return ucb
    
    def is_expanded(self):
        return True if self.children else False


    def expand(self, state, turn, action_probs):
        self.state = state
        self.turn = turn
        # print(action_probs)
        for a, prob in enumerate(action_probs):
            # print(a,prob)
            if prob != 0: 
                self.children[a] = Node(prior_prob=prob, turn=turn*-1)


class MCTS:
    def __init__(self, env, model, num_simulations):
        self.env = env
        self.model = model
        self.num_simulations = num_simulations

    def run(self, state, turn):
        root = Node(state=state, prior_prob=0, turn=turn)

        action_probs, _ = self.model.predict(state)
        valid_actions = self.env.get_valid_actions(state)
        # print(action_probs, valid_actions)
        action_probs = action_probs * valid_actions
        action_probs /= np.sum(action_probs)
        root.expand(state=state,turn=turn, action_probs=action_probs)

        for _ in range(self.num_simulations):
            node = root
            path = [node]

            while node.is_expanded():
                action, node = node.select_child()
                path.append(node)

            parent = path[-2]
            state = parent.state

            next_state, _ = self.env.get_next_state(state, action, turn)
            next_state = self.env.get_perspective_state(next_state, turn*-1)
            value = self.env.is_done(next_state, turn)

            if value is None:
                action_probs, value = self.model.predict(next_state)
                valid_actions = self.env.get_valid_actions(next_state)
                action_probs = action_probs * valid_actions  # mask invalid moves
                action_probs /= np.sum(action_probs)
                node.expand(next_state, parent.turn * -1, action_probs)

            self.backpropagate(path, value, parent.turn*-1)

        return root
    
    def backpropagate(self, path, value, turn):
        for node in reversed(path):
            node.value_sum += value if node.turn == turn else -value
            node.visit_count += 1


    




        