# NLU
Repository for the master course Natural Language Understanding assignments.

## Assignments

### LM
The objective is to implement various techniques that can enhance the performance of a simple language model for next-word prediction and to understand how each of these techniques influences the model. 
I have experimented with the model architecture, substituting the RNN with an LSTM block, using different optimizers, and applying different regularization techniques such as Dropout, Weight Tying, Variational Dropout, and Non-Monotonically Triggered Average SGD.

### NLU
The objective for this assignment is to jointly perform slot filling and intent prediction on the ATIS dataset. 
Intent prediction requires the model to classify the whole sequence to a predefined set of possible user intentions. 
Slot filling consists of identifying and classifying the relevant entities in the sequence, used to parameterize the user intent.
For the first part, we are tasked with implementing a functioning system using an LSTM backbone for the word token representations. For the second part, we are tasked with finetuning a pre-trained transformer BERT encoder to do the same task.

### SA
For this assignment, we are requested to Implement a model based on a Pre-trained Transformer model (such as BERT or RoBERTa) for the Aspect-based Sentiment Analysis task regarding the extraction of the aspect terms only. 
The dataset used is the Laptop partition of the SemEval2014 task 4.

## Execution
Each assignment contains a `main.py` file that can be executed to train the models or just evaluate the saved ones.
To execute the script you need to be in the same folder as the `main.py` file and run the following command:
```bash
  python main.py
```
The default behaviour is to only run the evaluation of the saved models in the `models` folder.
To train them, you can specify the flag `--Train` which will prompt the script to train and save the models in the `models` folder.


## Repository structure
```
  ├── LM             # first assignment
  |   ├── LM.pdf         # report for the first assignment
  |   └── ...        
  ├── SA             # second assignment
  |   ├── SA.pdf         # report for the second assignment
  |   └── ...        
  ├── NLU            # third assignment
  |   ├── NLU.pdf        # report for the third assignment
  |   └── ...        
  |
  |
  ├── .gitignore     # gitignore file
  ├── README.md      # this file
  └── LICENSE        # license file
```
