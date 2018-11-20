from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import random
import pandas as pd
import math
"""
data = load_iris()
X = data.data
y = data.target

"""
data = pd.read_table("/home/giovanni/Desktop/PyProjects/Machine Learning/Wine/winequality-red.csv", header = 0, sep = ";") #importing data and splitting it into features and label
features = ["alcohol", "sulphates", "citric acid", "density", "total sulfur dioxide", "volatile acidity"]
X = data.loc[:,features]
y = data.loc[:, "quality"]


X_train, X_test, y_train, y_test = train_test_split(X,y, random_state = 0) #creating training and testing set


hidden_layer_values = [] #initialising tuple of possible values for the "hidden_layer_sizes" parameter
for i in range(5,70,5):
    for j in range(5,70,5):
        hl_size = (i,j)
        hidden_layer_values.append(hl_size)

DNA = {"activation": ("identity", "logistic", "tanh", "relu"), "solver": ("lbfgs", "adam", "sgd"), "hidden_layer_sizes": hidden_layer_values} #dictionary containing the hyperparameters


def initial_population(individuals_nr, genes = DNA): #initialising a population that has random characteristics, chosen from the possible ones
    population = pd.DataFrame(columns = ["activation", "solver", "hidden_layer_sizes_1", "hidden_layer_sizes_2", "accuracy_score"])
    for i in range(individuals_nr):
        population.loc[i,"activation"] = genes["activation"][random.randrange(len(genes["activation"]))]
        population.loc[i,"solver"] = genes["solver"][random.randrange(len(genes["solver"]))]
        population.loc[i,"hidden_layer_sizes_1"] = genes["hidden_layer_sizes"][random.randrange(len(genes["hidden_layer_sizes"]))][0]
        population.loc[i,"hidden_layer_sizes_2"] = genes["hidden_layer_sizes"][random.randrange(len(genes["hidden_layer_sizes"]))][1]
    return population #returns DF of the population with the characteristics, and and empty column for the accuracy -> for evaluation of the classifier's performance


def train_n_test(population, X_train = X_train, y_train = y_train, X_test = X_test, y_test = y_test): #trains each element of the population and evaluates the accuracy_score, appending it to the last column 
    for i in range(population.shape[0]):
        cls = MLPClassifier(early_stopping= True, max_iter = 500, random_state = 0, activation = population.iloc[i,0], solver = population.iloc[i,1],
                            hidden_layer_sizes = (population.iloc[i,2], population.iloc[i,3]))
        cls.fit(X_train,y_train)
        y_predict = cls.predict(X_test)
        population.loc[i, "accuracy_score"] = accuracy_score(y_test, y_predict)        
    return population #returns DF of the population with characteristics (= to the input DF) and the additional accuracy_score column


def selection_n_breeding(population, fraction_best_kept = 0.5):
    population.sort_values(ascending = False, by = ["accuracy_score"], inplace = True)
    survivors = population.iloc[0:int(population.shape[0]*fraction_best_kept),:].reset_index(drop = True)

    mothers = survivors.iloc[survivors.index %2 == 0].reset_index(drop = True) #mothers = even rows of the population
    fathers = survivors.iloc[survivors.index %2 == 1].reset_index(drop = True) #fathers = uneven rows
    next_generation = pd.DataFrame(columns = ["activation", "solver", "hidden_layer_sizes_1", "hidden_layer_sizes_2"])
    
    for i in range(population.shape[0]):
        for j in range(population.shape[1]-1):
            rand_mom = random.randint(0,mothers.shape[0]-1)
            rand_dad = random.randint(0,fathers.shape[0]-1)
            next_generation.loc[i,next_generation.columns[j]] = random.choice([mothers.iloc[rand_mom,j],fathers.iloc[rand_dad,j]]) #choses randomy between mothers' and fathers' characteristics
            
            if (i+j)%11 == 0: #Randomness: every 11th i+j element (inclunding 0th) has a random mutation choosen fron the DNA pool
                #print("\nyeah, MUTATION!!!")
                #print("old value was", next_generation.iloc[i,j])
                if (next_generation.columns[j] == "hidden_layer_sizes_1") or (next_generation.columns[j] == "hidden_layer_sizes_2"):
                    next_generation.loc[i,next_generation.columns[j]] = DNA["hidden_layer_sizes"][random.randrange(len(DNA["hidden_layer_sizes"]))][random.randint(0,1)]
                else:
                    next_generation.loc[i,next_generation.columns[j]] = DNA[next_generation.columns[j]][random.randrange(len(DNA[next_generation.columns[j]]))]
                #print("new value is", next_generation.iloc[i,j])
                
    return survivors, next_generation, mothers, fathers #returns the upper "fraction_best_kept" of the population, filetring out the best performing characteristics

#survivors, next_generation, mothers, fathers = selection_n_breeding()


def evolution(generations = 5): #packs all of the previous functions together: 0 creates population; 1 selects the best and evaluates them; 2 splits in mothers and fathers and produces the offspring
    mean_score = []
    max_score = []
    population = initial_population(individuals_nr = 20) #this population has only the columns ["activation", "solver", "hidden_layer_sizes_1", "hidden_layer_sizes_2"] NO accuracy_score!!
    for i in range(generations):
        print("\ngeneration nr", i)
        trained_population = train_n_test(population) #returns the same population with the extra columns "accuracy_score"
        print("mean score", trained_population.accuracy_score.mean())
        print("max score", trained_population.accuracy_score.max())
        mean_score.append(trained_population.accuracy_score.mean())
        max_score.append(trained_population.accuracy_score.max())
        survivors, new_population, mothers, fathers = selection_n_breeding(trained_population) #selects the best of the trained_population, and returns a population based on their characteristics
        population = new_population
    print("\nuber generation")
    uber_population = train_n_test(population) #returns the same population with the extra columns "accuracy_score"
    mean_score.append(uber_population.accuracy_score.mean())
    max_score.append(uber_population.accuracy_score.max())
    print("uber mean score", uber_population.accuracy_score.mean())
    print("uber max score", uber_population.accuracy_score.max())
    plt.plot(mean_score, label = "mean score")
    plt.plot(max_score, label = "max score")
    plt.xlabel("generarion")
    plt.legend()
    plt.show()
    plt.savefig()
    
    return uber_population.accuracy_score

final_score = evolution("genetic_alg")

