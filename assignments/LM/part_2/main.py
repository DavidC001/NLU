# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

# Import everything from functions.py file
from functions import runExps, testModels
import argparse


if __name__ == "__main__":

    # define command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--Train', type=bool, default=False)
    parser.add_argument('--Test', type=bool, default=True)
    args = parser.parse_args()

    if args.Train:
        runExps()
    if args.Test:
        testModels()