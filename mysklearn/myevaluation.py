import math
import random
from typing import Callable
import mysklearn.myutils as myutils

def train_test_split(X, y, test_size=0.33, random_state=None, shuffle=True):
    """Split dataset into train and test sets (sublists) based on a test set size.

    Args:
        X(list of list of obj): The list of samples
            The shape of X is (n_samples, n_features)
        y(list of obj): The target y values (parallel to X)
            The shape of y is n_samples
        test_size(float or int): float for proportion of dataset to be in test set (e.g. 0.33 for a 2:1 split)
            or int for absolute number of instances to be in test set (e.g. 5 for 5 instances in test set)
        random_state(int): integer used for seeding a random number generator for reproducible results
        shuffle(bool): whether or not to randomize the order of the instances before splitting

    Returns:
        X_train(list of list of obj): The list of training samples
        X_test(list of list of obj): The list of testing samples
        y_train(list of obj): The list of target y values for training (parallel to X_train)
        y_test(list of obj): The list of target y values for testing (parallel to X_test)

    Note:
        Loosely based on sklearn's train_test_split(): https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    """
    num_instances = len(X)
    if random_state is not None:
       random.seed(random_state)

    if shuffle:
        # TODO: shuffle the rows in X and y before splitting
        # be sure to maintain the parallel order of X and y!!
        # note: the unit test for train_test_split() does not test
        # your use of random_state or shuffle, but you should still
        # implement this and check your work yourself
        indecies = myutils.shuffle_in_place(num_instances)
    else:
        indecies = [ i for i in range(num_instances) ]

    if isinstance(test_size, float):
        test_size = math.ceil(num_instances * test_size)
    split_index = num_instances - test_size
    train = indecies[:split_index]
    test = indecies[split_index:]

    return [ X[i] for i in train ], [ X[i] for i in test ], [ y[i] for i in train ], [ y[i] for i in test ]

def kfold_cross_validation(X, n_splits=5):
    """Split dataset into cross validation folds.

    Args:
        X(list of list of obj): The list of samples
            The shape of X is (n_samples, n_features)
        n_splits(int): Number of folds.

    Returns:
        X_train_folds(list of list of int): The list of training set indices for each fold
        X_test_folds(list of list of int): The list of testing set indices for each fold

    Notes:
        The first n_samples % n_splits folds have size n_samples // n_splits + 1,
            other folds have size n_samples // n_splits, where n_samples is the number of samples
            (e.g. 11 samples and 4 splits, the sizes of the 4 folds are 3, 3, 3, 2 samples)
        Loosely based on sklearn's KFold split(): https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
    """
    num_instances = len(X)
    x_test = [ [] for _ in range(n_splits) ]
    i = 0
    while i < num_instances:
        for j in range(n_splits):
            x_test[j].append(i)
            i += 1
            if i >= num_instances:
                break

    x_train = [ [] for _ in range(n_splits) ]
    for i in range(n_splits):
        for j in range(n_splits):
            if j != i:
                x_train[i].extend(x_test[j])
    return x_train, x_test

def stratified_kfold_cross_validation(X, y, n_splits=5):
    """Split dataset into stratified cross validation folds.

    Args:
        X(list of list of obj): The list of instances (samples).
            The shape of X is (n_samples, n_features)
        y(list of obj): The target y values (parallel to X).
            The shape of y is n_samples
        n_splits(int): Number of folds.

    Returns:
        X_train_folds(list of list of int): The list of training set indices for each fold.
        X_test_folds(list of list of int): The list of testing set indices for each fold.

    Notes:
        Loosely based on sklearn's StratifiedKFold split(): https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html#sklearn.model_selection.StratifiedKFold
    """
    groups, _ = myutils.stratify(y)
    x_train = [ [] for _ in range(n_splits) ]
    i = 0
    j = 0
    last_bin = -1
    for group in groups:
        group_size = len(group)
        i = 0
        j = last_bin + 1
        while i < group_size:
            while j < n_splits:
                x_train[j].append(group[i])
                last_bin = j
                i += 1
                j += 1
                if i >= group_size:
                    break
            j = 0

    x_test = [ [] for _ in range(n_splits) ]
    for i in range(n_splits):
        for j in range(n_splits):
            if j != i:
                x_test[i].extend(x_train[j])
    return x_test, x_train

def confusion_matrix(y_true, y_pred, labels):
    """Compute confusion matrix to evaluate the accuracy of a classification.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of str): The list of all possible target y labels used to index the matrix

    Returns:
        matrix(list of list of int): Confusion matrix whose i-th row and j-th column entry
            indicates the number of samples with true label being i-th class
            and predicted label being j-th class

    Notes:
        Loosely based on sklearn's confusion_matrix(): https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
    """
    str_labels = [ str(label) for label in labels ]
    size = len(str_labels)
    matrix = [ [ 0 for _ in range(size) ] for _ in range(size) ]

    for i in range(len(y_true)):
        actual_index = str_labels.index(str(y_true[i]))
        predicted_index = str_labels.index(str(y_pred[i]))
        matrix[actual_index][predicted_index] += 1

    return matrix

def get_splitter(split_func: Callable[..., 'tuple[list[list[int]], list[list[int]]]'], *split_func_args, **split_func_kwargs):
    a = myutils.bind(split_func, *split_func_args, **split_func_kwargs)
    return a

def cross_validate(splitter: 'Callable[[], tuple[list[list[int]], list[list[int]]]]', x_data, y_data, model, *more_models):
    models = [model, *more_models]
    results = [ [[], []] for _ in range(len(models)) ]
    X_train_folds, X_test_folds = splitter()
    for index in range(len(X_train_folds)):
        train_fold = X_train_folds[index]
        test_fold = X_test_folds[index]
        # prepare data for iteration
        X_train = [ x_data[i] for i in train_fold ]
        y_train = [ y_data[i]  for i in train_fold ]
        X_test =  [ x_data[i] for i in test_fold  ]
        y_test =  [ y_data[i]  for i in test_fold  ]
        # Predict
        for i, model in enumerate(models):
            model.fit(X_train, y_train)
            results[i][0].extend(y_test)
            results[i][1].extend(model.predict(X_test))
    return results
