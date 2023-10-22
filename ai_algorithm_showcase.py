import numpy as np
import os 
import copy
import random
from collections import defaultdict
from math import inf as infinity
import time

 
def load_matrix(matrix_file_name): # read and load the current matrix
    with open(matrix_file_name, 'r') as f:
        data = f.read()
        data2=data.replace('\n',',').split(',')
    matrix = np.zeros((5, 5))
    for i in range(5):
        for j in range(5):
            matrix[i,j]=int(data2[5*i+j])
    return matrix

def write_matrix(matrix, matrix_file_name_output): # wirte the new matrix into new txt file
    with open(matrix_file_name_output, 'w') as f:
        for i in range(5):
            for j in range(5):
                f.write(str(int(matrix[i,j])))
                if j<4:
                    f.write(',')
                if j==4:
                    f.write('\n')

def next_move_wolf(matrix): # random walk for wolf
    candidates=[]
    for i in range(5):
        for j in range(5):
            if matrix[i,j]==2:
                if i+1<5:
                    if matrix[i+1,j]==0:
                        candidates.append([i,j,i+1,j])
                if i-1>=0:
                    if matrix[i-1,j]==0:
                        candidates.append([i,j,i-1,j])
                if j+1<5:
                    if matrix[i,j+1]==0:
                        candidates.append([i,j,i,j+1])
                if j-1>=0:
                    if matrix[i,j-1]==0:
                        candidates.append([i,j,i,j-1])
                if i+2<5:
                    if matrix[i+2,j]==1 and matrix[i+1,j]==0:
                        candidates.append([i,j,i+2,j])
                if i-2>=0:
                    if matrix[i-2,j]==1 and matrix[i-1,j]==0:
                        candidates.append([i,j,i-2,j])
                if j+2<5:
                    if matrix[i,j+2]==1 and matrix[i,j+1]==0:
                        candidates.append([i,j,i,j+2])
                if j-2>=0:
                    if matrix[i,j-2]==1 and matrix[i,j-1]==0:
                        candidates.append([i,j,i,j-2])
    # move_idx=np.random.randint(0, len(candidates))
    return candidates

def next_move_sheep(matrix): # random walk for sheep
    candidates=[]
    for i in range(5):
        for j in range(5):
            if matrix[i,j]==1:
                if i+1<5:
                    if matrix[i+1,j]==0:
                        candidates.append([i,j,i+1,j])
                if i-1>=0:
                    if matrix[i-1,j]==0:
                        candidates.append([i,j,i-1,j])
                if j+1<5:
                    if matrix[i,j+1]==0:
                        candidates.append([i,j,i,j+1])
                if j-1>=0:
                    if matrix[i,j-1]==0:
                        candidates.append([i,j,i,j-1])
    # move_idx=np.random.randint(0, len(candidates))
    return candidates

def AIAlgorithm(filename, movemade): # a showcase for random walk
    iter_num=filename.split('/')[-1]
    iter_num=iter_num.split('.')[0]
    iter_num=int(iter_num.split('_')[1])
    matrix=load_matrix(filename)
    if movemade==False:
        maxDepth=2
        start_time = time.perf_counter()
        currentScore, currentMove=minimax(matrix,movemade,'sheep',5,0,-infinity,infinity)
        elapsed_time = time.perf_counter() - start_time
        [start_row, start_col, end_row, end_col]=currentMove
        matrix2=copy.deepcopy(matrix)
        matrix2[end_row, end_col]=1
        matrix2[start_row, start_col]=0    
    if movemade==True:
        start_time = time.perf_counter()
        currentScore, currentMove=minimax(matrix,movemade,'wolf',5,0,-infinity,infinity)
        elapsed_time = time.perf_counter() - start_time
        [start_row, start_col, end_row, end_col]=currentMove
        matrix2=copy.deepcopy(matrix)
        matrix2[end_row, end_col]=2
        matrix2[start_row, start_col]=0
        
    matrix_file_name_output=filename.replace('state_'+str(iter_num), 'state_'+str(iter_num+1)) 
    write_matrix(matrix2, matrix_file_name_output)

    return start_row, start_col, end_row, end_col

def minimax(matrix,movemade,player,maxDepth,currentDepth,alpha,beta):
    print(matrix)
    if len(next_move(matrix,movemade))==0 or currentDepth == maxDepth:
        return evaluate(matrix,player), None
    if currentPlayer(movemade)==player:
        bestScore = -infinity
    else:bestScore=infinity
    bestMove = None
    for move in next_move(matrix,movemade):
        new_matrix=apply_move(matrix,move,movemade)
        currentScore, currentMove=minimax(new_matrix,not movemade,player,maxDepth,currentDepth+1,alpha,beta)
        # alpha = max(alpha, bestScore)
        #     if beta <= alpha:
        if currentPlayer(movemade)==player:
            if currentScore>bestScore:
                bestScore=currentScore
                bestMove=move
            if beta <= bestScore:
                break  
            alpha = max(alpha, bestScore)
        else:
            if currentScore<bestScore:
                bestScore=currentScore
                bestMove=move
            if bestScore <= alpha:
                break  
            beta = min(beta, bestScore)
    print(bestMove,bestScore)
    return bestScore,bestMove

def evaluate(matrix,player):
    score=150
    count = np.count_nonzero(matrix == 1)
    if player=='sheep':
        for i in range(5):
            for j in range(5):
                if count>5:
                    if matrix[i,j]==1:
                        score+=1
                        if(i>1 and matrix[i-2,j]==2 and matrix[i-1,j]==0):
                            score-=3
                        if(i<3 and matrix[i+2,j]==2 and matrix[i+1,j]==0):
                            score-=3
                        if(j>1 and matrix[i,j-2]==2 and matrix[i,j-1]==0):
                            score-=3
                        if(j<3 and matrix[i,j+2]==2 and matrix[i,j+1]==0):
                            score-=3
                    if matrix[i,j]==2:
                        if(i>0 and matrix[i-1,j]==1):
                            score+=1
                            if i==0:
                                score+=1
                        if(i<4 and matrix[i+1,j]==1):
                            score+=1
                            if i==4:
                                score+=1
                        if(j>0 and matrix[i,j-1]==1):
                            score+=1
                            if j==0:
                                score+=1
                        if(j<4 and matrix[i,j+1]==1):
                            score+=1
                            if j==4:
                                score+=1

                else:
                    if(i>1 and matrix[i-2,j]==2 and matrix[i-1,j]==0):
                        score-=4
                    if(i<3 and matrix[i+2,j]==2 and matrix[i+1,j]==0):
                        score-=4
                    if(j>1 and matrix[i,j-2]==2 and matrix[i,j-1]==0):
                        score-=4
                    if(j<3 and matrix[i,j+2]==2 and matrix[i,j+1]==0):
                        score-=4
                    if matrix[i,j]==2:
                        if(i>0 and matrix[i-1,j]==1):
                            score+=1
                            if i==0:
                                score+=1
                        if(i<4 and matrix[i+1,j]==1):
                            score+=1
                            if i==4:
                                score+=1
                        if(j>0 and matrix[i,j-1]==1):
                            score+=1
                            if j==0:
                                score+=1
                        if(j<4 and matrix[i,j+1]==1):
                            score+=1
                            if j==4:
                                score+=1
                        
    else:
        for i in range(5):
            for j in range(5):
                if matrix[i,j]==2:
                    if(i>0 and matrix[i-1,j]==1):
                        score-=1
                        if i==0:
                            score-=1
                    if(j>0 and matrix[i,j-1]==1):
                        score-=1
                        if j==0:
                            score-=1
                    if(i<4 and matrix[i+1,j]==1):
                        score-=1
                        if i==4:
                            score-=1
                    if(j<4 and matrix[i,j+1]==1):
                        score-=1
                        if j==4:
                            score-=1
                    if(i>1 and matrix[i-2,j]==1):
                        score+=2
                    if(i<3 and matrix[i+2,j]==1):
                        score+=2
                    if(j>1 and matrix[i,j-2]==1):
                        score+=2
                    if(j<3 and matrix[i,j+2]==1):
                        score+=2
                    score-=count*3
    return score

def monte_carlo_search(matrix,num_simulations,movemade):
    # Get all legal moves for the current state
    if movemade==True:
        legal_moves = next_move_wolf(matrix)
    else:
        legal_moves=next_move_sheep(matrix)
        # Initialize dictionaries to store total reward and number of simulations for each move
    total_rewards = {}
    num_simulations_per_move = {}
    for move in legal_moves:
        total_rewards[tuple(move)] = 0
        num_simulations_per_move[tuple(move)] = 0
        # Run simulations
    for i in range(num_simulations):
        # Randomly choose a legal move
        move = random.choice(legal_moves)
        # Simulate the game from the chosen move
        reward = simulate_game(matrix, move,movemade)
        # Update total reward and number of simulations for the chosen move
        total_rewards[tuple(move)] += reward
        num_simulations_per_move[tuple(move)] += 1
        print(num_simulations_per_move[tuple(move)])
        # Calculate the average reward for each move
    avg_rewards = {}
    for move in (legal_moves+100):
        avg_rewards[tuple(move)] = total_rewards[tuple(move)] / num_simulations_per_move[tuple(move)]
        
        # Choose the move with the highest average reward
    best_move = max(avg_rewards, key=avg_rewards.get)
    return best_move

def simulate_game(matrix,move,movemade):
    # Make a copy of the current state
    new_matrix = copy.deepcopy(matrix)
    print(new_matrix)
    # Apply the move to the new state
    new_matrix = apply_move(new_matrix, move,movemade)
    print(new_matrix)
    # Alternate between wolves and sheep until the game is over
    while True:
        # Check if the wolves have won
        #check winning
        if movemade==True:
            if checkWinning(new_matrix)==1:   
                return 1  # Return a negative reward for wolves winning
            
            # Check if the sheep have won
            if checkWinning(new_matrix)==2:
                return -1  # Return a positive reward for sheep winning
        else:
            if checkWinning(new_matrix)==1:   
                return -1  # Return a negative reward for wolves winning
            
            # Check if the sheep have won
            if checkWinning(new_matrix)==2:
                return 1  # Return a positive reward for sheep winning
        # Get all legal moves for the current player
        legal_moves = next_move(new_matrix,movemade)
        
        # Check if there are any legal moves for the current player
        if not legal_moves:
            return 0  # Return 0 for a tie
        
        # Choose a random move for the current player
        move = random.choice(legal_moves)
        
        # Apply the move to the new state
        new_matrix = apply_move(new_matrix, move,movemade)

def currentPlayer(movemade):
    if(movemade==False):
        return 'sheep'
    else:
        return 'wolf'

def next_move(new_matrix,movemade):
    if movemade==True:
        legal_moves=next_move_wolf(new_matrix)
    else:
        legal_moves=next_move_sheep(new_matrix)
    return legal_moves


def apply_move(matrix,move,movemade):
    simulation_matrix=copy.deepcopy(matrix)
    if(movemade==True):
        simulation_matrix[move[2],move[3]] = 2
        simulation_matrix[move[0],move[1]] = 0
    else:
        simulation_matrix[move[2],move[3]] = 1
        simulation_matrix[move[0],move[1]] = 0
    return simulation_matrix
            

def checkWinning(matrix):
        # 1: wolf wins, 2: sheep wins, 0: continually gaming
        # check wolves winning
        sheep_num = 0
        wolf_neighbour = []
        wolf_win = True
        sheep_win = True
        winner = 0
        line_range = [0, 1, 2, 3, 4]
        for r in range(len(matrix)):
            for c in range(len(matrix[r])):
                if matrix[r][c] == '1':
                    sheep_num += 1
                elif matrix[r][c] == '2':
                    if r - 1 in line_range:
                        wolf_neighbour.append((r-1, c))
                    if r + 1 in line_range:
                        wolf_neighbour.append((r + 1, c))
                    if c - 1 in line_range:
                        wolf_neighbour.append((r, c - 1))
                    if c + 1 in line_range:
                        wolf_neighbour.append((r, c + 1))
                else:
                    pass
        for item in wolf_neighbour:
            if matrix[item[0]][item[1]] == '0':
                sheep_win = not sheep_win
                break
        if sheep_num > 2:
            wolf_win = not wolf_win

        if wolf_win:
            winner = 'wolf'
        if sheep_win:
            winner = 'sheep'

        return winner
# import numpy as np
# import os 
# import copy

# def load_matrix(matrix_file_name): # read and load the current state
#     with open(matrix_file_name, 'r') as f:
#         data = f.read()
#         data2=data.replace('\n',',').split(',')
#     matrix = np.zeros((5, 5))
#     for i in range(5):
#         for j in range(5):
#             matrix[i,j]=int(data2[5*i+j])
#     return matrix

# def write_matrix(matrix, matrix_file_name_output): # wirte the new state into new txt file
#     with open(matrix_file_name_output, 'w') as f:
#         for i in range(5):
#             for j in range(5):
#                 f.write(str(int(matrix[i,j])))
#                 if j<4:
#                     f.write(',')
#                 if j==4:
#                     f.write('\n')

# def next_move_wolf(matrix): # random walk for wolf
#     candidates=[]
#     for i in range(5):
#         for j in range(5):
#             if matrix[i,j]==2:
#                 if i+1<5:
#                     if matrix[i+1,j]==0:
#                         candidates.append([i,j,i+1,j])
#                 if i-1>=0:
#                     if matrix[i-1,j]==0:
#                         candidates.append([i,j,i-1,j])
#                 if j+1<5:
#                     if matrix[i,j+1]==0:
#                         candidates.append([i,j,i,j+1])
#                 if j-1>=0:
#                     if matrix[i,j-1]==0:
#                         candidates.append([i,j,i,j-1])
#                 if i+2<5:
#                     if matrix[i+2,j]==1 and matrix[i+1,j]==0:
#                         candidates.append([i,j,i+2,j])
#                 if i-2>=0:
#                     if matrix[i-2,j]==1 and matrix[i-1,j]==0:
#                         candidates.append([i,j,i-2,j])
#                 if j+2<5:
#                     if matrix[i,j+2]==1 and matrix[i,j+1]==0:
#                         candidates.append([i,j,i,j+2])
#                 if j-2>=0:
#                     if matrix[i,j-2]==1 and matrix[i,j-1]==0:
#                         candidates.append([i,j,i,j-2])
#     move_idx=np.random.randint(0, len(candidates))
#     return candidates[move_idx]

# def next_move_sheep(matrix): # random walk for sheep
#     candidates=[]
#     for i in range(5):
#         for j in range(5):
#             if matrix[i,j]==1:
#                 if i+1<5:
#                     if matrix[i+1,j]==0:
#                         candidates.append([i,j,i+1,j])
#                 if i-1>=0:
#                     if matrix[i-1,j]==0:
#                         candidates.append([i,j,i-1,j])
#                 if j+1<5:
#                     if matrix[i,j+1]==0:
#                         candidates.append([i,j,i,j+1])
#                 if j-1>=0:
#                     if matrix[i,j-1]==0:
#                         candidates.append([i,j,i,j-1])
#     move_idx=np.random.randint(0, len(candidates))



#     return candidates[move_idx]

# def ai_algorithm(filename, movemade): # a showcase for random walk
#     iter_num=filename.split('/')[-1]
#     iter_num=iter_num.split('.')[0]
#     iter_num=int(iter_num.split('_')[1])
#     matrix=load_matrix(filename)
#     if movemade==True:
#         [start_row, start_col, end_row, end_col]=next_move_wolf(matrix)
#         matrix2=copy.deepcopy(matrix)
#         matrix2[end_row, end_col]=2
#         matrix2[start_row, start_col]=0
            
#     if movemade==False:
#         [start_row, start_col, end_row, end_col]=next_move_sheep(matrix)
#         matrix2=copy.deepcopy(matrix)
#         matrix2[end_row, end_col]=1
#         matrix2[start_row, start_col]=0
        
#     matrix_file_name_output=filename.replace('state_'+str(iter_num), 'state_'+str(iter_num+1)) 
#     write_matrix(matrix2, matrix_file_name_output)

#     return start_row, start_col, end_row, end_col

