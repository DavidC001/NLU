# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

# Import everything from functions.py file
from functions import runExps, testModels

# Define the batch size
batch_size = 256


if __name__ == "__main__":
    run_exps = False
    if run_exps: runExps()
    
    testModels()