"""
Handle reading in the data and representing feature values
"""
import os
import numpy as np
from scipy.io import loadmat


class Schema(object):

    def __init__(self, ids, feature_names, is_nominal, nominal_values):
        self.ids = ids
        self.feature_names = feature_names
        self._is_nominal = is_nominal
        self.nominal_values = nominal_values

    def get_nominal_value(self, feature_index, value_index):
        if not self._is_nominal[feature_index]:
            raise ValueError('Feature %d is not nominal.' % feature_index)

        return self.nominal_values[feature_index][value_index]

    def is_nominal(self, feature_index):
        return self._is_nominal[feature_index]


def get_dataset(dataset_name, base_directory='.'):
    """
    Loads a dataset with the given name. The associated `.mat` file
    must be in the directory `base_directory`.
    @param dataset_name : name of `.mat` file holding dataset
    @param base_directory : location of `.mat` file holding dataset
    @return (Schema, X, y) : X is a examples-by-features sized NumPy array,
                             and y is a 1-D array of associated -1/+1 labels
    """
    mat = loadmat(os.path.join(base_directory, dataset_name),
                  appendmat=True, chars_as_strings=True, squeeze_me=True)

    feature_names = [str(s) for s in mat['feature_names']]
    is_nominal = [bool(b) for b in mat['is_nominal']]
    nominal_values = [[str(s) for s in values]
                      for values in mat['nominal_values']]

    ids = [str(s) for s in mat['ids']]
    X = mat['examples']
    y = mat['labels']

    schema = Schema(ids, feature_names, is_nominal, nominal_values)

    return schema, X, y
