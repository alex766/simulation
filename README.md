# Simulation

The code needed to run the simulation exists in `run.py`.

## Choice Model Parameters

The parameters for different choice models can be read in with the $\texttt{parse\_params}$ function. The function takes in the file name and returns a dictionary of model names as keys with a dictionary of parameters as values. 

## Running the Code
The function $\texttt{run\_and\_plot}$ takes in the different tunable parameters for the simulation, including the ratio of the choice model parameters excluding the booking constant, the ratio of the booking constant, and other factors. It plots the acceptance probabilities and average number of accepted loads over 5 runs of each pricing algorithm. 