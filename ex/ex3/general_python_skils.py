# Write a Python function that takes a 
# sequence of numbers and determines whether 
# all the numbers are different from each other.
# Testing seq: [1, 4, 3, 4, 1] -> False
# 	       [1, 2, 3, 4, 5] -> True

def check_if_unique(the_list):


    new_list = list(set(the_list))
    if len(new_list) < len(the_list):
        return False
    return True

# Write a Python program to print a long text, 
# convert the string to a list and print all the words 
# and their frequencies. 

def word_freq(sentence):
    sentence_list = sentence.split(' ')
    u_sentence_dict = dict.fromkeys(set(sentence_list) , 0)
    k = list(set(sentence_list))
    for word in sentence_list:
        u_sentence_dict[word] += 1
    return u_sentence_dict

# Write a Python program which accepts 
# a sequence of comma-separated numbers from user 
# and generate a list with those numbers.

def sep_camma(camma_numbers):
    if not type(camma_numbers) == str:
        return "Error : waiting for a string"
    
    num_list = camma_numbers.split(',')
    for index , num in enumerate(num_list):
        try:
            num_list[index] = int(num)  
        except:
            return "Error : One of the values could not be cast to int"

    return num_list

# Write a Python script to add a key to a dictionary.

def add_key_to_dict(dictionary , add_key):
    dictionary[add_key] = "defualt"
    return dictionary

# Checking if a key is in a dictionary

def is_key_in_dict(dictionary , key):
    if key in dictionary:
        return True
    return False

# Creating a dictionary with keys from 1 to 15
# and values from 1**2 (the power of 2) to 15**2

def dumb_function():
    """
    This function prints a dictionary with keys from 1 to 15
    and value from 1^2 to 15^2
    """
    dictionary = {}

    for i in range(1,16):
        import math
        dictionary[i] = i**2
    
    return dictionary

# Function that returns the sum of a list
# or the multiplication of the list,
# depending on the argument passed under op

def op_on_list(num_list , op = "sum"):
    tot = 0
    
    if op == "sum":
        for n in num_list:
            tot += n
    if op == "mul":
        for n in num_list:
            tot *= n
    
    return tot

# This function checks whether a 
# number is in the range between 
# two other numbers

def is_in_range(num : int , range : list , include_bottom = False , include_top = False) -> bool:
    range.sort()


    if len(range) == 2:
        if include_bottom:
            con_1 = num >= range[0]
        else:
            con_1 = num > range[0]
        if include_top:
            con_2 = num <= range[1]
        else:
            con_2 = num < range[1]

        if con_1 and con_2:
            return True
        return False

# This function checks whether all 
# the numbers in a certain list are 
# in a certain range of numbers
def list_in_range(num_list : list , range : list) -> bool:

    for n in num_list:
        if not is_in_range(n , range):
            return False

    return True

# This function prints only the even 
# numbers in a list of numbers

def print_only(num_list : list , num_type = "even"):
    for n in num_list:
        if num_type == "even":
            con = n % 2 == 0
        elif num_type == "all":
            con = True
        else:
            con = not n % 2 == 0 
        if con:
            print(n)




