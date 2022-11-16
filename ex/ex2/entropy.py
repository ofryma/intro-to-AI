from math import log


def calculate_entropy(num_true_results , num_false_results):
    """
    This function calculate entropy for true false situation
    """

    
    base = 2
    tot_results = num_false_results + num_true_results
    p1 = num_true_results/tot_results
    p2 = num_false_results/tot_results
    entropy = p1*log(p1 , base) + p2*log(p2,base)
    entropy *= -1

    return entropy

def calculate_atribute_entropy(num_true_results , num_false_results , num_of_subjects):
    """
    This function calculate the entropy of a single atribute
    """
    
    tot_results = num_false_results + num_true_results
    x = tot_results/num_of_subjects
    entropy = x * calculate_entropy(num_true_results , num_false_results)
    return entropy

def calculate_information_gain(num_true_results , num_false_results , atr_entropy):
    """
    This function calculate the information gain for a given atribute entropy
    """    
    ig = calculate_entropy(num_true_results , num_false_results) - atr_entropy
    return ig

def calculate_intrinsic_info(list_number_of_atributes):
    """
    This function calculate the intrinsic info
    the list given is a list of integers representing the 
    number of subject in each atribute
    """

    num_of_subjects = sum(list_number_of_atributes)
    base = 2
    intrinsic_info = 0
    for num in list_number_of_atributes:
        x = num/num_of_subjects
        intrinsic_info += x*log(x,base)
    
    intrinsic_info *= -1

    return intrinsic_info

def calculate_gain_ratio(informatin_gain , intrinsic_info):
    return informatin_gain/intrinsic_info




sick = 16
not_sick = 14

entropy_1 = calculate_entropy(sick,not_sick)
print("Entropy : " ,entropy_1)

num_of_subjects = sick + not_sick
child_entropy = calculate_atribute_entropy(7,4,num_of_subjects)
adult_entropy = calculate_atribute_entropy(2,5,num_of_subjects)
elderly_entropy = calculate_atribute_entropy(7,5,num_of_subjects)

atributes_entropy = child_entropy + adult_entropy + elderly_entropy

information_gain = calculate_information_gain(sick , not_sick , atributes_entropy)
print("Information Gain (age) : " , information_gain)

intrinsic_info = calculate_intrinsic_info([11,7,12])
print("Intrinsic Information : " , intrinsic_info)

gain_ratio = calculate_gain_ratio(information_gain , intrinsic_info)
print("Gain Ratio : " , gain_ratio)



