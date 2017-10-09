
# coding: utf-8

# In[16]:

import numpy as np
from copy import deepcopy
import heapq as hq
import os.path
import json


# In[ ]:

Sub1Goal = ([[1.0, 2.0, 99.0, 99.0], [5.0, 6.0, 99.0, 99.0], [99.0, 99.0, 99.0, 99.0], [99.0, 99.0, 99.0, 0.0]])
Sub2Goal = ([[99.0, 99.0, 3.0, 4.0], [99.0, 99.0, 7.0, 8.0], [99.0, 99.0, 99.0, 99.0], [99.0, 99.0, 99.0, 0.0]])
Sub3Goal = ([[99.0, 99.0, 99.0, 99.0], [99.0, 99.0, 99.0, 99.0], [9.0, 10.0, 99.0, 99.0], [13.0, 14.0, 99.0, 0.0]])
Sub4Goal = ([[99.0, 99.0, 99.0, 99.0], [99.0, 99.0, 99.0, 99.0], [99.0, 99.0, 11.0, 12.0], [99.0, 99.0, 15.0, 0.0]])


# In[3]:

def KeyEncodeC(State, l):
    StateList = (np.array(State)).flatten()
    Non99 = []
    if l == 1:
        poslist = (0,1,2,5,6)
    elif l == 2:
        poslist = (0,3,4,7,8)
    elif l == 3:
        poslist = (0,9,10,13,14)
    else:
        poslist = (0,11,12,15)
    for i in poslist:
        Non99.append((np.where(StateList == i))[0][0])
    return ' '.join([str(i) for i in Non99])

def KeyEncodeD(State,l):
    StateList = (np.array(State)).flatten()
    Non99 = []
    if l == 1:
        poslist = (1,2,5,6)
    elif l == 2:
        poslist = (3,4,7,8)
    elif l == 3:
        poslist = (9,10,13,14)
    else:
        poslist = (11,12,15)
    for i in poslist:
        Non99.append((np.where(StateList == i))[0][0])
    return ' '.join([str(i) for i in Non99])

def DBGenerator(GoalState, l):
    fringe = [(GoalState, 0)]
    closed = {KeyEncodeC(GoalState,l):0}
    Subdb = {KeyEncodeD(GoalState,l):0}

    while len(fringe) > 0: 
        PopSubState = fringe.pop(0)
        for s in Successor(PopSubState[0]):
            if KeyEncodeC(s[0],l) not in closed:
                fringe.append((s[0],PopSubState[1]+1))
                closed[KeyEncodeC(s[0],l)] = 0
                if KeyEncodeD(s[0],l) not in Subdb:
                    Subdb[KeyEncodeD(s[0],l)] = PopSubState[1]+1 
    return Subdb


# In[18]:

json.dump(DBGenerator(Sub1Goal, 1), open("db1.txt",'w'))
json.dump(DBGenerator(Sub2Goal, 2), open("db2.txt",'w'))
json.dump(DBGenerator(Sub3Goal, 3), open("db3.txt",'w'))
json.dump(DBGenerator(Sub4Goal, 4), open("db4.txt",'w'))
Sub1DB = json.load(open("db1.txt"))
Sub2DB = json.load(open("db2.txt"))
Sub3DB = json.load(open("db3.txt"))
Sub4DB = json.load(open("db4.txt"))


# In[28]:

# A* Approach to solve 15-puzzle problem:

# Importing necessary packages:
import numpy as np
import copy
from copy import deepcopy
import heapq as hq
import os.path
import json 
import sys

# Loading the input text file as a 4x4 Numpy array:
def ReadFile(InputFile):
    InitialState = np.loadtxt(InputFile)
    return InitialState

# Returns the position of '0' in the puzzle:
def LocateZero(State): 
    position = zip(*np.where(State == 0))
    return position

# Returns the successors for a given configuration of the puzzle and moves done to achieve it:
def Successor(State):
    State = np.array(State)
    Successors = []
    r,c = LocateZero(State)[0] 
    if (c+1 <= 3):
        Move = "L1" + str(r+1)
        trans_array = deepcopy(State)
        trans_array[r,c], trans_array[r,c+1] =  trans_array[r,c+1], trans_array[r,c]
        Successors.append((trans_array.tolist(),Move))
        if (c+2 <= 3):
            Move = "L2" + str(r+1)
            trans_array[r,c+1], trans_array[r,c+2] =  trans_array[r,c+2], trans_array[r,c+1]
            Successors.append((trans_array.tolist(),Move))
            if (c+3 <= 3):
                Move = "L3" + str(r+1)
                trans_array[r,c+2], trans_array[r,c+3] =  trans_array[r,c+3], trans_array[r,c+2]
                Successors.append((trans_array.tolist(),Move))
    if (c-1 >= 0):
        Move = "R1" + str(r+1)
        trans_array = deepcopy(State)
        trans_array[r,c], trans_array[r,c-1] =  trans_array[r,c-1], trans_array[r,c]
        Successors.append((trans_array.tolist(),Move))
        if (c-2 >=0):
            Move = "R2" + str(r+1)
            trans_array[r,c-1], trans_array[r,c-2] =  trans_array[r,c-2], trans_array[r,c-1]
            Successors.append((trans_array.tolist(),Move))
            if (c-3 >= 0):
                Move = "R3" + str(r+1)
                trans_array[r,c-2], trans_array[r,c-3] =  trans_array[r,c-3], trans_array[r,c-2]
                Successors.append((trans_array.tolist(),Move))
    if (r+1 <= 3):
        Move = "U1" + str(c+1)
        trans_array = deepcopy(State)
        trans_array[r,c], trans_array[r+1,c] =  trans_array[r+1,c], trans_array[r,c]
        Successors.append((trans_array.tolist(),Move))
        if (r+2 <= 3):
            Move = "U2" + str(c+1)
            trans_array[r+1,c], trans_array[r+2,c] =  trans_array[r+2,c], trans_array[r+1,c]
            Successors.append((trans_array.tolist(),Move))
            if (r+3 <= 3):
                Move = "U3" + str(c+1)
                trans_array[r+2,c], trans_array[r+3,c] =  trans_array[r+3,c], trans_array[r+2,c]
                Successors.append((trans_array.tolist(),Move))
    if (r-1 >= 0):
        Move = "D1" + str(c+1)
        trans_array = deepcopy(State)
        trans_array[r,c], trans_array[r-1,c] =  trans_array[r-1,c], trans_array[r,c]
        Successors.append((trans_array.tolist(),Move))
        if (r-2 >=0):
            Move = "D2" + str(c+1)
            trans_array[r-1,c], trans_array[r-2,c] =  trans_array[r-2,c], trans_array[r-1,c]
            Successors.append((trans_array.tolist(),Move))
            if (r-3 >= 0):
                Move = "D3" + str(c+1)
                trans_array[r-2,c], trans_array[r-3,c] =  trans_array[r-3,c], trans_array[r-2,c]
                Successors.append((trans_array.tolist(),Move))
    return Successors

# Returns the heuristic based on the pre-computed costs in the database:
def Heuristic(State):
    k1 = KeyEncodeD(State,1)
    k2 = KeyEncodeD(State,2)
    k3 = KeyEncodeD(State,3)
    k4 = KeyEncodeD(State,4)
    if k1 in Sub1DB:
        HofS1 = Sub1DB[k1]
    if k2 in Sub2DB:
        HofS2 = Sub2DB[k2]
    if k3 in Sub3DB:
        HofS3 = Sub3DB[k3]
    if k4 in Sub4DB:
        HofS4 = Sub4DB[k4]
    HofS = max(HofS1, HofS2, HofS3, HofS4)
    return HofS

# Determines parity to check if a given puzzle has a solution:
def Parity(InitialState):
    N = LocateZero(InitialState)[0][0] + 1    
    StateList = [val for sublist in InitialState for val in sublist if val != 0]
    for i in range(len(StateList)):
        for j in range((i+1),len(StateList)) :
            if StateList[i] > StateList[j]:
                N += 1
    return (N % 2) 

# Solves the puzzle by employing priority queue:
def solve(InitialState):
    if Parity(InitialState) == 1:
        return False
    if Heuristic(InitialState) == 0:
        return InitialState
    fringe = []
    closed = []
    hq.heappush(fringe,(Heuristic(InitialState), 0, "IS",InitialState.tolist()))
    while len(fringe) > 0:
        PopState = hq.heappop(fringe)#
        GofS = PopState[1]+1
        Path = PopState[2]
        for s in Successor(PopState[3]):
            if s[0] not in closed:
                hq.heappush(fringe, (Heuristic(s[0]) + GofS, GofS, Path +' '+ s[1], s[0]))
                closed.append(s[0])
            if Heuristic(s[0]) == 0:
                return (Path + ' '+ s[1])
    return False

# Goal states for four sub-problems:
Sub1Goal = ([[1.0, 2.0, 99.0, 99.0], [5.0, 6.0, 99.0, 99.0], [99.0, 99.0, 99.0, 99.0], [99.0, 99.0, 99.0, 0.0]])
Sub2Goal = ([[99.0, 99.0, 3.0, 4.0], [99.0, 99.0, 7.0, 8.0], [99.0, 99.0, 99.0, 99.0], [99.0, 99.0, 99.0, 0.0]])
Sub3Goal = ([[99.0, 99.0, 99.0, 99.0], [99.0, 99.0, 99.0, 99.0], [9.0, 10.0, 99.0, 99.0], [13.0, 14.0, 99.0, 0.0]])
Sub4Goal = ([[99.0, 99.0, 99.0, 99.0], [99.0, 99.0, 99.0, 99.0], [99.0, 99.0, 11.0, 12.0], [99.0, 99.0, 15.0, 0.0]])

# Returns position of Zero and the other 4 nos. in the subproblem:

def KeyEncodeC(State, l):
    StateList = (np.array(State)).flatten()
    Non99 = []
    if l == 1:
        poslist = (0,1,2,5,6)
    elif l == 2:
        poslist = (0,3,4,7,8)
    elif l == 3:
        poslist = (0,9,10,13,14)
    else:
        poslist = (0,11,12,15)
    for i in poslist:
        Non99.append((np.where(StateList == i))[0][0])
    return ' '.join([str(i) for i in Non99])

# Returns position of the 4 nos. in the subproblem:
def KeyEncodeD(State,l):
    StateList = (np.array(State)).flatten()
    Non99 = []
    if l == 1:
        poslist = (1,2,5,6)
    elif l == 2:
        poslist = (3,4,7,8)
    elif l == 3:
        poslist = (9,10,13,14)
    else:
        poslist = (11,12,15)
    for i in poslist:
        Non99.append((np.where(StateList == i))[0][0])
    return ' '.join([str(i) for i in Non99])

# Returns the database with the cost for each subproblem:
def DBGenerator(GoalState, l):
    fringe = [(GoalState, 0)]
    closed = {KeyEncodeC(GoalState,l):0}
    Subdb = {KeyEncodeD(GoalState,l):0}

    while len(fringe) > 0: 
        PopSubState = fringe.pop(0)
        for s in Successor(PopSubState[0]):
            if KeyEncodeC(s[0],l) not in closed:
                fringe.append((s[0],PopSubState[1]+1))
                closed[KeyEncodeC(s[0],l)] = 0
                if KeyEncodeD(s[0],l) not in Subdb:
                    Subdb[KeyEncodeD(s[0],l)] = PopSubState[1]+1 
    return Subdb

# Creates the database if the DB files are not already available in the working directory:
if (os.path.isfile('db1.txt') != True ):
	json.dump(DBGenerator(Sub1Goal, 1), open("db1.txt",'w'))
	json.dump(DBGenerator(Sub2Goal, 2), open("db2.txt",'w'))
	json.dump(DBGenerator(Sub3Goal, 3), open("db3.txt",'w'))
	json.dump(DBGenerator(Sub4Goal, 4), open("db4.txt",'w'))

# Loads the DB files into dictionaries:
Sub1DB = json.load(open("db1.txt"))
Sub2DB = json.load(open("db2.txt"))
Sub3DB = json.load(open("db3.txt"))
Sub4DB = json.load(open("db4.txt"))


# In[29]:

# Print the output for a given input puzzle:
InputFile = 'C:\Users\shahi\OneDrive\Sem 3\AI\Assignments\Assignment 1\Puzzle_15_ip.txt' 
InitialState = ReadFile(InputFile)
Result = solve(InitialState) 
print(Result.replace(Result[:3],''))

