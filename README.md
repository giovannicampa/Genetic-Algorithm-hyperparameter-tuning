# Genetic-Algorithm-hyperparameter-tuning
Chosing the best combination of Hyperparameters for a MLP Classifier through a Genetic Algorithm

The values that the Hyperparameters can be are stored in the dictionary DNA.
A population with random characteristics (Hyperparameter combinations) is initialised randomly and stored in a DataFrame.
The performance of each "individual" of the population is mesaured (accuracy_score) and saved in the DF previously created.
Then the population is reduced and the best n% of it is allowed to reproduce, transferring its charcateristics (the Hyperparameters) to the offspring. Here also a random mutation happens, that allows for unseen characteristics from the DNA to show up in the population.
The evealuation of the performance, the selection of the best, and the reproduction are done a nr of times (generations).
The results are shown in a graph.
