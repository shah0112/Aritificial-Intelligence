# A* Approach to solve 15-puzzle problem:
# COLLABORATED WITH TEAM MEMBERS CHETAN PATIL AND ADITYA ARIGELA ABOUT APPROACH. DISCUSSED CONCEPT OF PATTERN DATABASES WITH SRIRAM SITHARAMAN
'''
>> Description of the search problem was formulated: 
state space: S = {4x4 array containing 0-15 in any position, in any order}
Successor function: Succ = 
Edge weights : We = d, where d is the depth of the node in the search tree. For the successors of initial state, We = 1 
and for their successors We = 2. Similarly, it in incremeneted in steps of 1 as the depth increases
Goal state: Sg = [ 1  2  3  4
                   5  6  7  8
				   9  10 11 12
				   13 14 15 0 ]
Heuristic function: H(s) = Cost taken to reach goal state of sub-problem without considering the other tiles in the puzzle.
The mail problem of achieving the goal state mentioned above is split into 4 subproblems with the following 4 sub goal states:
[ 1 2 X X  [ X X 3 4   [ X  X  X X   [ X X X  X
  5 6 X X    X X 7 8     X  X  X X     X X X  X
  X X X X    X X X X     9  10 X X     X X 11 12
  X X X X]   X X X X]    13 14 X X]    X X 15 0 ]
where 'X' can be any number from 0 to 15 that is not already in the puzzle board.
Using the same successor function, a tree is constructed with the above goal states as the root. For the given puzzle, 4 sub boards
are created with positions of [0,1,2,5,6]; [0,3,4,7,8]; [0,9,10,13,14]; [0,11,12,15] maintained the same in puzzle and remaining tiles 
substituted by '99'. The depth of the resultant board in the tree created will be taken as the heuristic. This heuristic will always 
be admissible because, it is the cost taken to reach a sub-goal considering all other tiles do not affect this process. But in reality 
other tiles will increase the number of steps taken to achieve these sub-goals.

>> Brief description of how your search algorithm works
Step 1: Create 4 DBs with pre-computed costs for sub-problems/ import pre-created DB files
Step 2: The puzzle is read from the text file and stored as a 4x4 numpy array. Goal state is defined
Step 3: If parity for the puzzle is different from parity of goal state, return Failure
Step 4: Generate successors of the current state by sliding 1/2/3 tiles L/R/U/D depending on the position of '0'
Step 5: Check if one of the successors is the goal state, else push the successors into priority queue with the priority as the sum of 
	    Edge weight and heuristic
Step 6: Pop the item with highest priority and repeat steps 4 & 5

>> Discussion of problems faced, simplifications, and design decisions 
Problem: The main problem faced was deciding on an admissible heuristic for the problem. Since more than 1 tile can be slid in a single move, the 
usual heuristic functions like Euclidean distance, Manhattan distance, no. of misplaced tiles etc. failed to be admissible. Few other 
heuristic functions were too slow in finding the optimal solution though they were admissible. In order to find the optimal solution in 
short time, Database with pre-computed costs were created. This proved to be closest to the actual cost when compared to other functions 
that were considered. 
Design decisions: Dictionaries were used since they tend to be fast when compared to list of lists. Since the key of a dictionary cannot be a list,
the positions of the required numbers were used as keys. This also made comparison faster and easier. 
'''

''' IMPORTANT NOTE: PLEASE IMPORT THE DATABASE FILES db1.txt, db2.txt, db3.txt, db4.txt BEFORE RUNNING THE CODE'''
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

# Print the output for a given input puzzle:

InputFile = sys.argv[1] 
InitialState = ReadFile(InputFile)
Result = solve(InitialState) 
print(Result.replace(Result[:3],''))