
# coding: utf-8

# In[168]:

#Calculating the score of a die roll:
def roll_sum(roll):
    return 25 if ( roll[0] == roll[1] and roll[0] == roll[2] ) else sum(roll)

#Calculating the expected score of a re-roll:
def exp_sum(listofrolls):
    return (sum(roll_sum(r) for r in listofrolls)*1.0/len(listofrolls))

#Getting the list of all possible re-rolls:
def best_reroll(roll):
    reroll_list = {}
    reroll_list['1'] = exp_sum(list([i,roll[1],roll[2]] for i in range(1,7))) 
    reroll_list['2'] = exp_sum(list([roll[0],i,roll[2]] for i in range(1,7)))
    reroll_list['3'] = exp_sum(list([roll[0],roll[1],i] for i in range(1,7)))
    reroll_list['1, 2'] = exp_sum(list([i,j,roll[2]] for i in range(1,7) for j in range(1,7)))
    reroll_list['1, 3'] = exp_sum(list([i,roll[1],j] for i in range(1,7) for j in range(1,7)))
    reroll_list['2, 3'] = exp_sum(list([roll[0],i,j] for i in range(1,7) for j in range(1,7)))
    reroll_list['1, 2, 3'] = exp_sum(list([i,j,k] for i in range(1,7) for j in range(1,7) for k in range(1,7)))

    key = max(reroll_list, key = reroll_list.get)
    return(key, reroll_list[key])


# In[171]:

#giving input die values:
dice =  [int(n) for n in raw_input('Enter number on each of the three dice separated by space: ').split()]
rollpos, roll_exp_sum = best_reroll(dice)

#printing the output:
if roll_sum(dice) == 25:
    print("Do not Re-roll! You have the best score already")
elif roll_exp_sum > roll_sum(dice):
    print("Reroll dice " + rollpos + ", the expected score is " + str(roll_exp_sum) )
else:
    print("Do not Re-roll! The score won't get any better")

