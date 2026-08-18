# BASE-AI
A Base program for creating ML models. 

This program was created to make creating simple Machine Learning projects. 

It consists of a backend and a middle end. Will create a front-end, not done yet. 

I am just starting out on github, so any contribution, constructive feedback, issues, bug-reports, or feature suggestions are welcome. This is my first actual project on github, so I don't really know the community, so any help is welcome. Thanks!!

Please note: The model can only output whole numbers from -32768 to 32767


# Goals
The goals of this project are to make simple machine learning projects more accessable for people. The project is intended to eventually be plug and play, with many different training types available, including pattern recognition, bot dev, NLP, GAN, transformers, diffusion etc. Also include a library of Pre-trained models that can be specialized. 


# Functions
Middle end: train and generation;

These two functions are the lifeblood of this project. They take a reward function and create a model. 

Backend: 

new(input_len, output_len, hidden_layers) # returns a new randomly configured model

input_len = number of model inputs (What the model can see)

output_len = number of model outputs (What the model can do)

hidden_layers = number of intermediate layers of decision making that the model can do ex. [5, 5, 5]

mutate(model) # returns a slightly different version of the model, for reinforcment learning

run(inputs[], model) # returns a list of the models outputs

save(model, file_path) # saves a model to a file, recommend using .pt for file extension

load(file_path) # returns the model at the path

