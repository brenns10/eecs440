EECS 440 Programming Assignment 4
=================================

Instructions
------------

In addition to those functions you already have filled in previous programming
assignments (you need to modify them to handle weighted instances if necessary),
please fill in the following new functions:

fit(), predict(), predict_proba() in bagger.py/Bagger.m, booster.py/Booster.m


Please run the framework from the command line as follows, for example:

for python:
    python main.py --dataset_directory data --dataset voting --meta_algorithm boosting --meta_iters 20  dtree --depth 2 

for matlab:
    main('dataset_directory', 'data', 'dataset', 'voting', 'meta_algorithm', 'boosting', 'meta_iters', 20, 'classifier', 'dtree', 'depth', 2)
