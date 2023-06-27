import numpy as np
from operator import add, neg

def solve(payoff_matrix, iterations=1000):
    payoff_matrix = np.array(payoff_matrix)
    '''Return the oddments (mixed strategy ratios) for a given payoff matrix'''
    transpose = list(zip(*payoff_matrix))
    numrows = len(payoff_matrix)
    numcols = len(transpose)
    row_cum_payoff = [0] * numrows
    col_cum_payoff = [0] * numcols
    colpos = list(range(numcols))
    rowpos = list(map(neg, range(numrows)))
    colcnt = [0] * numcols
    rowcnt = [0] * numrows
    active = 0
    # print("payoff_matrix:",payoff_matrix)
    # print("transpose:",transpose)
    # print("numrows:",numrows)
    # print("numcols:",numcols)
    # print("row_cum_payoff:",row_cum_payoff)
    # print("col_cum_payoff:",col_cum_payoff)
    # print("colpos:",colpos)
    # print("rowpos:",rowpos)
    # print("colcnt:",colcnt)
    # print("rowcnt:",rowcnt)
    # print("active:",active)
    for i in range(iterations):
        rowcnt[active] += 1     

        col_cum_payoff = list(map(add, payoff_matrix[active], col_cum_payoff))
        # print("col_cum_payoff:",col_cum_payoff)

        active = min(list(zip(col_cum_payoff, colpos)))[1]
        # print("active:",active)

        colcnt[active] += 1   

        row_cum_payoff = list(map(add, transpose[active], row_cum_payoff))
        # print("row_cum_payoff:",row_cum_payoff)

        active = -max(list(zip(row_cum_payoff, rowpos)))[1]
        # print("active:",active)
        

    value_of_game = (max(row_cum_payoff) + min(col_cum_payoff)) / 2.0 / iterations
    return rowcnt, colcnt, value_of_game

def solve2(payoff_matrix, iterations=50):
    payoff_matrix = np.array(payoff_matrix)
    '''Return the oddments (mixed strategy ratios) for a given payoff matrix'''
    transpose_payoff = np.transpose(payoff_matrix)
    row_cum_payoff = np.zeros(len(payoff_matrix))
    col_cum_payoff = np.zeros(len(transpose_payoff))

    col_pos = np.arange(len(transpose_payoff))
    row_pos = (-np.arange(len(payoff_matrix))).tolist()

    col_count = np.zeros(len(transpose_payoff))
    row_count = np.zeros(len(payoff_matrix))
    active = 0
    # print("payoff_matrix:",payoff_matrix)
    # print("transpose payoff:",transpose_payoff)
    # print("row_cum_payoff:",row_cum_payoff)
    # print("col_cum_payoff:",col_cum_payoff)
    # print("colpos:",col_pos)
    # print("rowpos:",row_pos)
    # print("colcnt:",col_count)
    # print("rowcnt:",row_count)
    # print("active:",active)

    for i in range(iterations):
        row_count[active] += 1 
        col_cum_payoff += payoff_matrix[active]
        # print("col_cum_payoff:",col_cum_payoff)

        active = np.argmin(col_cum_payoff)
        # print("active:",active)

        col_count[active] += 1 

        row_cum_payoff += transpose_payoff[active]
        # print("row_cum_payoff:",row_cum_payoff)
        
        active = np.argmax(row_cum_payoff)
        # print("active:",active)
        
        
    value_of_game = (max(row_cum_payoff) + min(col_cum_payoff)) / 2.0 / iterations  
    row_prob = row_count / iterations
    col_prob = col_count / iterations
    
    return row_prob, col_prob, value_of_game


# print(solve2([[2,3,1,4], [1,2,5,4], [2,3,4,1], [4,2,2,2]]))    # Example on page 185
# print(solve([[4,0,2], [6,7,1]]))   
print(solve2([[4,0,2], [6,7,1]]))                  # Exercise 2 number 3
# print(solve([[ 0.25873908,  0.6761097,   0.14118922,  0.5469872,   0.23603989,  0.15997642],
#  [ 0.76323545,  0.80511963,  0.6657765,   0.57493603,  0.63344216,  0.7027939 ],
#  [ 0.19144577,  0.21063896,  0.43571037,  0.30484983,  0.40354544,  0.37116015],
#  [ 0.5546464,   0.35572556,  0.56326634,  0.26714608,  0.47160277,  0.55380416],
#  [ 0.07642518,  0.14801878, -0.06580116, -0.1657206,   0.10791554,  0.1657201 ],
#  [ 0.00458195, -0.13565752, -0.31278306,  0.07323447,  0.08669062,  0.20530091]]))
# for i in range(100):
#     print(i,solve2([[ 0.25873908,  0.6761097,   0.14118922,  0.5469872,   0.23603989,  0.15997642],
#     [ 0.76323545,  0.80511963,  0.6657765,   0.57493603,  0.63344216,  0.7027939 ],
#     [ 0.19144577,  0.21063896,  0.43571037,  0.30484983,  0.40354544,  0.37116015],
#     [ 0.5546464,   0.35572556,  0.56326634,  0.26714608,  0.47160277,  0.55380416],
#     [ 0.07642518,  0.14801878, -0.06580116, -0.1657206,   0.10791554,  0.1657201 ],
#     [ 0.00458195, -0.13565752, -0.31278306,  0.07323447,  0.08669062,  0.20530091]], iterations=i)[2])