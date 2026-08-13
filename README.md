# BASE-AI
A Base program for creating ML models. 

This program was created to make creating simple Machine Learning projects more simple. 

It consists of 5 parts: 

new(input_len, output_len, hidden_layers) # returns a new randomly configured model

input_len = number of model inputs (What the model can see)

output_len = number of model outputs (What the model can do)

hidden_layers = number of intermediate layers of decision making that the model can do ex. [5, 5, 5]

mutate(model) # returns a slightly different version of the model, for reinforcment learning

run(inputs[], model) # returns a list of the models outputs

save(model, file_path) # saves a model to a file, recommend using .pt for file extension

load(file_path) # returns the model at the path
