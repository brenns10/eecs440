Please fill in the following functions:

fit(), predict(), predict_proba() from class ArtificialNeuralNetwork in ann.py/ArtificialNeuralNetwork.m

accuracy(), recall(), precision(), auc() in stats.py/StatisticsManager.m (reuse as needed from previous assignments)

get_folds() in main.py/get_folds.m (reuse from previous assignments)

Please run the framework from the command line as follows, for example:

for python:
python main.py --dataset_directory data --dataset voting ann --gamma 0.1 --num_hidden 40 --layer_sizes 1 --max_iters 500 

for matlab:
main('dataset_directory', 'data', 'dataset', 'voting', 'classifier', 'ann', 'gamma', 0.1, 'num_hidden', 40, 'layer_sizes', 1, 'max_iters', 500)
