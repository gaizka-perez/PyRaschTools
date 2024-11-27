import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import math



def perc_correct(person):
    total_correct = sum(person)
    total_items = len(person)
    percentage = total_correct / total_items * 100
    return(percentage)


#This chunk calculates the percentage of correct items that each person got and viceversa, removes perfect items or persons and iterates until no items or persons are needed to be removed    
def remove_unusable_scores(data):
    data = pd.DataFrame(data)
    while True:
        correct_percentages_p = []
        correct_percentages_i = []

        items_removed = 0
        persons_removed = 0

        for item in data.columns:
            item_difficulty = perc_correct(data.loc[:,item])
            if item_difficulty == 100 or item_difficulty == 0:
                items_removed += 1
                data = data.drop(item, axis=1)

        for person in data.index:
            person_ability = perc_correct(data.loc[person])
            if person_ability == 100 or person_ability == 0:
                persons_removed += 1
                data = data.drop(person, axis=0)

        if items_removed == 0 and persons_removed == 0:
            break 

    return(data) 

def estimates(data):
    
    #Calculate estimates based on PROX estimation method
    difficulty_logit_init = []
    for item in data.columns:
        prop_correct = perc_correct(data.loc[:,item]) / 100
        difficulty_logit_init.append(np.log((1-prop_correct) / prop_correct))
        
    difficulty_logit_init_cent = []   
    for logit in range(len(difficulty_logit_init)):
        difficulty_logit_init_cent.append(difficulty_logit_init[logit] - np.mean(difficulty_logit_init))

    U = np.std(difficulty_logit_init) ** 2

    ability_logit_init = []
    for person in range(data.shape[0]):
        prop_correct = perc_correct(data.iloc[person]) / 100
        ability_logit_init.append(np.log(prop_correct / (1-prop_correct))) 

    V = np.std(ability_logit_init) ** 2

    ab_exp_factor = ((1+U/2.80) / (1-U*V/8.35)) ** (1/2)

    diff_exp_factor = (1+V/2.89) / (1-U*V/8.35)

    difficulty_logit_calib = [i * diff_exp_factor for i in difficulty_logit_init_cent]

    ability_logit_calib = [i * ab_exp_factor for i in ability_logit_init]

    #Calculate standard errors
    std_err_diff = []
    for item in range(data.shape[1]):
        std_err_diff.append(diff_exp_factor*(data.shape[0]/sum(data.iloc[:,item])*(data.shape[0]-sum(data.iloc[:,item])))**(1/2))

    std_err_abil = []
    for person in range(data.shape[0]):
        std_err_abil.append(ab_exp_factor*(data.shape[1]/sum(data.iloc[person])*(data.shape[1]-sum(data.iloc[person])))**(1/2))

    return(difficulty_logit_calib, ability_logit_calib, std_err_diff, std_err_abil)


def item_fit(data, items, people):
    #Compute the expected, variance and standardize residual score for each item for each person and store in a list
    expected = {}
    variance = {}
    standardize_residual ={}

    for person in range(people.shape[0]):

        list_name = f"{people.index[person]}" 
        expected[list_name] = []

        list_name_var = f"{people.index[person]}" 
        variance[list_name_var] = []

        list_name_std_res = f"{people.index[person]}" 
        standardize_residual[list_name_std_res] = []       

        for item in range(items.shape[0]):
            expected_value = np.exp(people.iloc[person,0]- items.iloc[item,0]) / (1+ np.exp(people.iloc[person,0]- items.iloc[item,0]))
            variance_value = expected_value*(1-expected_value)
            actual_response = data.iloc[person,item]

            standardize_residual_value = (actual_response - expected_value) / np.sqrt(variance_value)

            expected[list_name].append(expected_value)
            variance[list_name_var].append(variance_value)
            standardize_residual[list_name_std_res].append(standardize_residual_value)


    # Convert dictionary to DataFrame and transpose it 

    expected_df = pd.DataFrame.from_dict(expected, orient='index')

    expected_df.columns = data.columns

    variance_df = pd.DataFrame.from_dict(variance, orient='index')

    variance_df.columns = data.columns

    standardize_residual_df = pd.DataFrame.from_dict(standardize_residual, orient='index')

    standardize_residual_df.columns = data.columns


    #Compute Outfit

    N = people.shape[0]

    item_outfits_meansq = []

    for item in range(items.shape[0]):
        std_residuald_sum_of_squares = (standardize_residual_df.iloc[:,item] ** 2).sum()

        item_outfit = std_residuald_sum_of_squares / N

        item_outfits_meansq.append(item_outfit)

    
    #Compute Infit
    
    item_infits_meansq = []

    for item in range(items.shape[0]):
        variance_std_residuald_sum_of_squares = (variance_df.iloc[:,item] * (standardize_residual_df.iloc[:,item] ** 2)).sum()

        variance_sum = (variance_df.iloc[:,item]).sum()

        item_infit = variance_std_residuald_sum_of_squares / variance_sum

        item_infits_meansq.append(item_infit)

    return(item_outfits_meansq, item_infits_meansq)





def dichotomous(data):
    data = remove_unusable_scores(data)
    difficulty_par, ability_par, std_err_diff, std_err_abil = estimates(data)

   

    item_names = data.columns
    person_names = list(data.index)
    difficulty_df_int = pd.DataFrame(
        {"difficulty_par" : difficulty_par, 
        "difficulty_std_err" : std_err_diff}, 
        index = item_names)
    
    ability_df_int = pd.DataFrame(
        {"ability_par" : ability_par, 
        "ability_std_err" : std_err_abil}, 
        index = person_names)
    

    outfit, infit = item_fit(data, difficulty_df_int, ability_df_int)

    difficulty_df = pd.DataFrame(
        {"difficulty_par" : difficulty_par, 
        "difficulty_std_err" : std_err_diff, 
        "Outfit_meansq" : outfit,
        "Infit_meansq" : infit},
        index = item_names)
    
    ability_df = pd.DataFrame(
        {"ability_par" : ability_par, 
        "ability_std_err" : std_err_abil}, 
        index = person_names)
    
    
    return (difficulty_df, ability_df)


### For Rasch visualizations

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def ICC(items):
    num_of_lines = items.shape[0]
    color=iter(cm.cool(np.linspace(0,1,num_of_lines)))

    for item in range(items.shape[0]):
        difficulty = items.iloc[item, 0]
        c=next(color) #Change colour for each line in plot
        x = np.linspace(-100, 100, 1000)  # 1000 points between -10 and 10
        x = x 
        # Calculate the sigmoid values
        y = sigmoid(x-difficulty) 
        # Plot the graph
        item_n = items.index[item]
        plt.plot(x, y, c=c, label = f"Item {item_n}")
  
    # Add labels and title
    plt.legend()
    plt.xlabel("Theta")
    plt.ylabel("Probability of correct response")
    plt.title("Item Characteristic Curve")
    # Set limits slightly bigger than range of function
    plt.xlim([-3, 5])
    plt.ylim([-0.1, 1.1])
    # Grid
    plt.grid(True)
    # Show the plot
    plt.show()


def item_info(theta, difficulty):
  return np.exp(theta-difficulty)/(1 + np.exp(theta-difficulty))**2


def IIF(items):
    num_of_lines = items.shape[0]
    color=iter(cm.cool(np.linspace(0,1,num_of_lines)))

    for item in range(items.shape[0]):
        difficulty = items.iloc[item, 0]
        c=next(color) #Change colour for each line in plot
        x = np.linspace(-100, 100, 1000)  # 1000 points between -10 and 10
        x = x 
        # Calculate the sigmoid values
        y = item_info(x,difficulty) 
        # Plot the graph
        item_n = items.index[item]
        plt.plot(x, y, c=c, label = f"Item {item_n}")
  
    # Add labels and title
    plt.legend()
    plt.xlabel("Theta")
    plt.ylabel("Item Information")
    plt.title("Item Information Function")
    # Set limits slightly bigger than range of function
    plt.xlim([-3, 5])
    plt.ylim([-0.1, 1.1])
    # Grid
    plt.grid(True)
    # Show the plot
    plt.show()


def TIF(items):
    x = np.linspace(-100, 100, 1000)  # 1000 points between -10 and 10
    y = np.zeros_like(x)  # Initialize y as an array of zeros with the same shape as x

    for item in range(items.shape[0]):
        x = x 
        # Calculate the sigmoid values
        y_item = item_info(x,items.iloc[item,0]) 
        y += y_item  # Sum the sigmoid values

    # Plot the graph
    plt.plot(x, y)
    # Add labels and title
    plt.xlabel("Theta")
    plt.ylabel("Information")
    plt.title("Test Information Function")
    # Set limits slightly bigger than range of function
    plt.xlim([-8, 9])
    plt.ylim([-0.1, y.max() + 0.1])
    # Grid
    plt.grid(True)
    # Show the plot
    plt.show() 



def generate_sample(num_people, num_items):
    np.random.seed(42)  # For reproducibility
    # Generate random 0s and 1s for the DataFrame
    data = np.random.randint(0, 2, size=(num_people, num_items))
    # Create the DataFrame
    df = pd.DataFrame(data, columns=[f'Item{i+1}' for i in range(num_items)])
    # Optionally, add row names (people)
    df.index = [f'Person{i+1}' for i in range(num_people)]
    return df

