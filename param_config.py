from ray import tune

# Which hyperparameters to search
param_config = {

"lr": 5e-4,
"num_train": 8000,
"num_valid": 1000,
"num_test": 1000,
"num_channels": [10],
"batch_size": 25
}

     