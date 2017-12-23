
# coding: utf-8

# In[6]:

#!/usr/bin/env python
# nrooks.py : Solve the N-Rooks problem!
# D. Crandall, 2016
# Updated by Zehua Zhang, 2017
# Optimized and included N-Queens - Shahidhya, 2017
# The N-Rooks problem is: Given an empty NxN chessboard, place N rooks on the board so that no rooks
# can take any other, i.e. such that no two rooks share the same row or column.
# The N-Queens problem is: Given an empty NxN chessboard, place N queens on the board so that no queens
# can take any other, i.e. such that no two queens share the same row, column or diagonal.

# Importing the required packages:
import sys
import numpy as np
from copy import deepcopy

# Count no. of pieces in given row:
def count_on_row(board, row):
    return sum(board[row]) 

# Count no. of pieces in given column:
def count_on_col(board, col):
    return sum([row[col] for row in board]) 

# Check if any diagonal has more than 1 Queen:
def count_on_diag(board):
    a = np.array(board)
    a1 = a[::-1,:]
    for i in range(N-1,-N,-1):
        if sum(a.diagonal(i)) > 1:
            return False
    for i in range(-N+1,N):
        if sum(a1.diagonal(i)) > 1:
            return False
    return True

# Count total no. of pieces on board:
def count_pieces(board):
    return sum([ sum(row) for row in board ] )

# Return a string with the board rendered in a human-friendly format:
def printable_board(board):
    boardX = deepcopy(board)
    boardX[negr-1][negc-1] = 99
    boarde = deepcopy(boardX)
    boarde = list(row[:-1] for row in boarde[:-1])
    if negr != 0 and negc != 0:
        if higher == 1:
            return "\n".join([ " ".join([ "R" if (col == 1 and ptype == "nrook") else "Q" if (col == 1 and ptype == "nqueen") else "X" if col == 99 else "_" for col in row]) for row in boarde])
        else:
            return "\n".join([ " ".join([ "R" if (col == 1 and ptype == "nrook") else "Q" if (col == 1 and ptype == "nqueen") else "X" if col == 99 else "_" for col in row]) for row in boardX])
    else:
        if higher == 1:
            return "\n".join([ " ".join([ "R" if (col == 1 and ptype == "nrook") else "Q" if (col == 1 and ptype == "nqueen") else "_" for col in row]) for row in boarde])
        else:
            return "\n".join([ " ".join([ "R" if (col == 1 and ptype == "nrook") else "Q" if (col == 1 and ptype == "nqueen") else "_" for col in row]) for row in board])                      

# Add a piece to the board at the given position, and return a new board:
def add_piece(board, row, col):
    return board[0:row] + [board[row][0:col] + [1,] + board[row][col+1:]] + board[row+1:]

# Get list of successors of given board state:
def successors(board):
    row_i = [i for i, x in enumerate(list(count_on_row(board, row) for row in range(0,N))) if x != 1]
    col_i = [i for i, x in enumerate(list(count_on_col(board, col) for col in range(0,N))) if x != 1]
    actuals = [add_piece(board, r, c) for r in row_i for c in col_i if (r == negr-1 and c == negc-1) != 1]
    if ptype == "nrook":
        return actuals
    else:
        Final = [item for item in actuals if count_on_diag(item)] 
        return Final

# Check if the board is a goal state:
def is_goal(board):
    return count_pieces(board) == N

# Solve n-rooksand n-queens:
def solve(initial_board):
    fringe = [initial_board]
    while len(fringe) > 0:
        for s in successors( fringe.pop() ):
            if is_goal(s):
                return(s)
            fringe.append(s)
    return False

# Input to the code: nrook/nqueen, n, unavailable position
ptype = "nqueen" # (sys.argv[1]) # nrook nqueen 
N =  10 # int(sys.argv[2])
negr = 3 # int(sys.argv[3])
negc = 9 # int(sys.argv[4])

if (N % 2) == 0:
    N = N+1
    higher = 1
else:
    higher = 0
rows = cols = N
initial_board =  [[0 for j in range(cols)] for i in range(rows)]
print ("Starting from initial board:\n" + printable_board(initial_board) + "\n\nLooking for solution...\n")
solution = solve(initial_board)
print (printable_board(solution) if solution else "Sorry, no solution found. :(")

