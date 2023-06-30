
import torch
import math
import random
import numpy as np

# env.py 의 ConnectFourEnv와는 다르게 게임 규칙만 들어가 있음 
# player는 1과 -1로 이루어져 있으므로 정규화할 필요 없음 
# self play 이므로 초기 player를 랜덤으로 둘 필요 없음 
# action은 one-hot encoding 된 상태로 받음 ex. [0,0,1,0,0,0,0]

## env.py로 옮김