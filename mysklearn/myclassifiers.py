import copy
import functools
from typing import Any
import random
import mysklearn.myutils as myutils

class MySimpleLinearRegressor:
    """Represents a simple linear regressor.

    Attributes:
        slope(float): m in the equation y = mx + b
        intercept(float): b in the equation y = mx + b

    Notes:
        Loosely based on sklearn's LinearRegression: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
        Terminology: instance = sample = row and attribute = feature = column
    """

    def __init__(self, slope=None, intercept=None):
        """Initializer for MySimpleLinearRegressor.

        Args:
            slope(float): m in the equation y = mx + b (None if to be computed with fit())
            intercept(float): b in the equation y = mx + b (None if to be computed with fit())
        """
        self.slope = slope
        self.intercept = intercept

    def fit(self, X_train, y_train):
        """Fits a simple linear regression line to X_train and y_train.

        Args:
            X_train(list of list of numeric vals): The list of training samples
                The shape of X_train is (n_train_samples, n_features)
                Note that n_features for simple regression is 1, so each sample is a list
                    with one element e.g. [[0], [1], [2]]
            y_train(list of numeric vals): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples
        """
        x_unpacked = [ feature[0] for feature in X_train ]
        mean_x = myutils.avg(x_unpacked)
        mean_y = myutils.avg(y_train)

        self.slope = sum([(myutils.avg(X_train[i]) - mean_x) * (y_train[i] - mean_y) for i in range(len(X_train))]) / sum([(myutils.avg(X_train[i]) - mean_x) ** 2 for i in range(len(X_train))])
        self.intercept = mean_y - self.slope * mean_x

    def predict(self, X_test):
        """Makes predictions for test samples in X_test.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)
                Note that n_features for simple regression is 1, so each sample is a list
                    with one element e.g. [[0], [1], [2]]

        Returns:
            y_predicted(list of numeric vals): The predicted target y values (parallel to X_test)
        """
        y_predicted = [ feature[0] * self.slope + self.intercept for feature in X_test]
        return y_predicted


class MyKNeighborsClassifier:
    """Represents a simple k nearest neighbors classifier.

    Attributes:
        n_neighbors(int): number of k neighbors
        X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train).
            The shape of y_train is n_samples

    Notes:
        Loosely based on sklearn's KNeighborsClassifier: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
        Assumes data has been properly normalized before use.
    """
    def __init__(self, n_neighbors=3):
        """Initializer for MyKNeighborsClassifier.

        Args:
            n_neighbors(int): number of k neighbors
        """
        self.n_neighbors = n_neighbors
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        """Fits a kNN classifier to X_train and y_train.

        Args:
            X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since kNN is a lazy learning algorithm, this method just stores X_train and y_train
        """
        self.X_train = X_train
        self.y_train = y_train

    def kneighbors(self, X_test):
        """Determines the k closes neighbors of each test instance.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            distances(list of list of float): 2D list of k nearest neighbor distances
                for each instance in X_test
            neighbor_indices(list of list of int): 2D list of k nearest neighbor
                indices in X_train (parallel to distances)
        """
        distances = []
        indecies = []
        normalized_train = myutils.normalize_table(self.X_train)
        normalized_test = myutils.normalize_table(X_test, against=self.X_train)
        for instance in normalized_test:
            inst_indecies = [i for i in range(len(self.X_train))]
            inst_distances = [myutils.distance(record, instance) for record in normalized_train]
            inst_distances, inst_indecies = myutils.sort_in_parallel(inst_distances, inst_indecies)
            distances.append(inst_distances[:self.n_neighbors])
            indecies.append(inst_indecies[:self.n_neighbors])
        return distances, indecies

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        _, neighbor_indecies = self.kneighbors(X_test)
        y_neighbors = [[self.y_train[index] for index in instance_indecies] for instance_indecies in neighbor_indecies]
        y_predicted = []
        for neighbors in y_neighbors:
            groups, classes = myutils.stratify(neighbors)
            group_sizes = [len(group) for group in groups]
            max_size = 0
            max_index = 0
            for i in range(len(group_sizes)):
                if group_sizes[i] > max_size:
                    max_size = group_sizes[i]
                    max_index = i
            y_predicted.append(classes[max_index])
        return y_predicted

class MyNaiveBayesClassifier:
    """Represents a Naive Bayes classifier.

    Attributes:
        X_train(list of list of obj): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train).
            The shape of y_train is n_samples
        priors(YOU CHOOSE THE MOST APPROPRIATE TYPE): The prior probabilities computed for each
            label in the training set.
        posteriors(YOU CHOOSE THE MOST APPROPRIATE TYPE): The posterior probabilities computed for each
            attribute value/label pair in the training set.

    Notes:
        Loosely based on sklearn's Naive Bayes classifiers: https://scikit-learn.org/stable/modules/naive_bayes.html
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self):
        """Initializer for MyNaiveBayesClassifier.

        """
        self.X_train = None
        self.y_train = None
        self.priors = None
        self.posteriors = None

    def fit(self, X_train, y_train):
        """Fits a Naive Bayes classifier to X_train and y_train.

        Args:
            X_train(list of list of obj): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since Naive Bayes is an eager learning algorithm, this method computes the prior probabilities
                and the posterior probabilities for the training data.
            You are free to choose the most appropriate data structures for storing the priors
                and posteriors.
        """
        self.X_train = X_train
        self.y_train = y_train

        width = len(X_train[0])
        labels = [ { v for v in myutils.get_column(X_train, i) } for i in range(width) ]
        class_labels = { c for c in y_train }
        class_label_counts = { c_l: 0.0 for c_l in class_labels }
        inst_count = float(len(X_train))
        priors = { class_label: [ { label: 0.0 for label in label_set } for label_set in labels ] for class_label in class_labels }

        for i in range(len(X_train)):
            class_label = y_train[i]
            for j in range(len(X_train[i])):
                attr = X_train[i][j]
                priors[class_label][j][attr] = priors[class_label][j][attr] + 1
            class_label_counts[class_label] = class_label_counts[class_label] + 1

        self.priors = { class_label: count / inst_count for class_label, count in class_label_counts.items() }
        self.posteriors = { class_label: [ { label: priors[class_label][i][label] / class_label_counts[class_label] for label in attr } for i, attr in enumerate(inst) ] for class_label, inst in priors.items() }

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        def getVal(a: float, b: "list[int, Any]", posteriors: "list[dict[Any, float]]")-> float:
            try:
                return a * posteriors[b[0]][b[1]]
            except:
                return a
        def posteriorFunc(posteriors: "list[dict[Any, float]]"):
            return lambda a, b: getVal(a, b, posteriors)

        return [ myutils.dict_max_key({ class_label: functools.reduce(posteriorFunc(posteriors), list(enumerate(X)), 1.0) * self.priors[class_label] for class_label, posteriors in self.posteriors.items() }) for X in X_test ]

class MyZeroRClassifier:
    """Represents a Zero-R classifier.

    Attributes:
        X_train(list of list of obj): The list of training instances (samples).
            The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train).
            The shape of y_train is n_samples
        prediction(obj): The most frequent value in y_train
    """
    def __init__(self):
        """Initializer for MyZeroRClassifier.

        """
        self.X_train = None
        self.y_train = None
        self.prediction = None

    def fit(self, X_train, y_train):
        """Fits a Zero-R classifier to X_train and y_train.

        Args:
            X_train(list of list of obj): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        """
        self.X_train = X_train
        self.y_train = y_train
        labels = set(y_train)
        tallies = { label: 0 for label in labels }
        for label in y_train:
            tallies[label] = tallies[label] + 1
        self.prediction = copy.deepcopy(myutils.dict_max_key(tallies))

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        return [ copy.deepcopy(self.prediction) for _ in X_test ]

class MyRandomClassifier:
    """Represents a Random classifier.

    Attributes:
        X_train(list of list of obj): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train).
            The shape of y_train is n_samples
    """
    def __init__(self):
        """Initializer for MyRandomClassifier.

        """
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        """Fits a Random classifier to X_train and y_train.

        Args:
            X_train(list of list of obj): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        """
        self.X_train = X_train
        self.y_train = y_train


    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        return [ self.y_train[random.randrange(0, len(self.y_train))] for _ in X_test ]

class MyDecisionTreeClassifier:
    """Represents a decision tree classifier.

    Attributes:
        X_train(list of list of obj): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train).
            The shape of y_train is n_samples
        tree(nested list): The extracted tree model.

    Notes:
        Loosely based on sklearn's DecisionTreeClassifier: https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self):
        """Initializer for MyDecisionTreeClassifier.

        """
        self.X_train = None
        self.y_train = None
        self.tree = None

    def select_attribute(self, instances: "list[list]", attribute_indexes: "list[int]", attribute_domains: "list[list]", class_index: int)-> int:
        leni = len(instances) * 1.0
        entropies = {}
        for i, index in enumerate(attribute_indexes):
            partition = [ self.compute_partition_stats([ instance for instance in instances if instance[index] == attribute_value ], class_index).values() for attribute_value in attribute_domains[index] ]
            e = [ (sum(v)/leni)*myutils.entropy(*v, total=sum(v)) for v in partition ]
            entropies[i] = sum(e)
        split_index = myutils.dict_min_key(entropies)
        return split_index

    def partition_instances(self, instances: "list[list]", attribute_index: int, attribute_domains: "list[list]")-> "dict[Any, list[list]]":
        # this is a group by attribute_index's domain, not by
        # the values of this attribute in instances
        attribute_domain = attribute_domains[attribute_index]
        return { attribute_value: [ instance for instance in instances if instance[attribute_index] == attribute_value ] for attribute_value in attribute_domain }

    def all_same_class(self, instances: "list[list]", class_index: int)-> bool:
        label = instances[0][class_index]
        for instance in instances:
            if instance[class_index] != label:
                return False
        return True

    def all_leaf_agree(self, nodes)-> bool:
        if nodes[0][0] != "Value" or nodes[0][2][0] != "Leaf" :
            return False
        label = nodes[0][2][1]
        for node in nodes:
            if node[2][0] != "Leaf":
                return False
            if node[2][1] != label:
                return False
        # print("Collapsing matching leafs")
        return True


    def compute_partition_stats(self, instances: "list[list]", class_index: int)-> "dict[Any, int]":
        """Return a list of stats: [[label1, occ1, tot1], [label2, occ2, tot2], ...]"""
        stats = { label: 0 for label in set(myutils.get_column(instances, class_index))}
        for instance in instances:
            stats[instance[class_index]] = stats[instance[class_index]] + 1
        return stats

    def tdidt(self, instances: "list[list]", attribute_indexes: "list[int]", attribute_domains: "list[list]", class_index: int)-> list:
        # select an attribute to split on
        attribute_index_index = self.select_attribute(instances, attribute_indexes, attribute_domains, class_index)
        attribute_index = attribute_indexes[attribute_index_index]
        attribute_indexes.pop(attribute_index_index)
        # cannot split on the same attribute twice in a branch
        tree = ["Attribute", attribute_index]
        # group data by attribute domains (creates pairwise disjoint partitions)
        partitions = self.partition_instances(instances, attribute_index, attribute_domains)
        # leaf_candidates = []
        for attribute_value, partition in partitions.items():
            value_subtree = ["Value", attribute_value]
            # print(tree[:2], value_subtree[:2],":",self.compute_partition_stats(partition, class_index))
            if len(partition) == 0:
                # print("CASE 3")
                # print(tree[:2], value_subtree[:2])
                #  CASE 3: no more instances to partition (empty partition) => backtrack and replace attribute node with majority vote leaf node
                stats = self.compute_partition_stats(instances, class_index)
                label = myutils.dict_max_key(stats)
                # value_subtree.append(["Leaf", label, stats[label], len(instances)])
                # tree.append(value_subtree)
                return ["Leaf", label, stats[label], len(instances)]
            elif self.all_same_class(partition, class_index):
                # print("CASE 1")
                #  CASE 1: all class labels of the partition are the same => make a leaf node
                value_subtree.append(["Leaf", partition[0][class_index], len(partition), len(instances)])
                tree.append(value_subtree)
            elif len(attribute_indexes) == 0:
                # print("CASE 2")
                #  CASE 2: no more attributes to select (clash) => handle clash w/majority vote leaf node
                stats = self.compute_partition_stats(partition, class_index)
                label = myutils.dict_max_key(stats)
                # leaf_candidates.append(["Leaf", label, stats[label], len(partition)])
                value_subtree.append(["Leaf", label, stats[label], len(partition)])
                tree.append(value_subtree)
            else:
                #  Recursive Case
                subtree = self.tdidt(partition, attribute_indexes.copy(), attribute_domains, class_index)
                value_subtree.append(subtree)
                tree.append(value_subtree)
        # if len(leaf_candidates) > 0:
            # leaf_index = myutils.dict_max_key({ i: leaf_candidate[2] for i, leaf_candidate in enumerate(leaf_candidates) })
            # return leaf_candidates[leaf_index]
        # print("t:",tree)
        if self.all_leaf_agree(tree[2:]):
            return ["Leaf", tree[2][2][1], sum(node[2][2] for node in tree[2:]), len(instances)]

        return tree

    def fit(self, X_train: "list[list]", y_train: list)-> None:
        """Fits a decision tree classifier to X_train and y_train using the TDIDT (top down induction of decision tree) algorithm.

        Args:
            X_train(list of list of obj): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since TDIDT is an eager learning algorithm, this method builds a decision tree model
                from the training data.
            Build a decision tree using the nested list representation described in class.
            Store the tree in the tree attribute.
            Use attribute indexes to construct default attribute names (e.g. "att0", "att1", ...).
        """
        self.X_train = X_train
        self.y_train = y_train
        width = len(X_train[0])
        # prepare indecies
        attribute_indexes = [ i for i in range(width) ]
        # stitch together X_train and y_train per Gina advice
        train = [ X_train[i] + [y_train[i]] for i in range(len(X_train)) ]
        # prepare domains, add class label domains column to match 'train' table and keep parameters simple
        attribute_domains = [ list(set(myutils.get_column(X_train, i))) for i in range(width) ]
        attribute_domains.append(list(set(y_train)))
        # initial call to tdidt current instances is the whole table (train)
        self.tree = self.tdidt(train, attribute_indexes.copy(), attribute_domains, width)

    def predict(self, X_test: "list[list]")-> list:
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        # self.print_decision_rules(["class", "age", "sex"], "survived")
        # print()
        y_predicted = []
        for instance in X_test:
            node = self.tree
            while node[0] != "Leaf":
                # n_start = node
                val = instance[node[1]]
                for n in node[2:]:
                    if n[0] != "Value":
                        break
                    if n[1] == val:
                        node = n[2]
                        break
                # if node == n_start:
                #     print("node:", node)
                #     print("instance:", instance)
                #     exit(1)
            y_predicted.append(node[1])
        return y_predicted

    def print_decision_rules(self, attribute_names:"list[str]" = None, class_name:str = "class")-> str:
        """Prints the decision rules from the tree in the format "IF att == val AND ... THEN class = label", one rule on each line.

        Args:
            attribute_names(list of str or None): A list of attribute names to use in the decision rules
                (None if a list is not provided and the default attribute names based on indexes (e.g. "att0", "att1", ...) should be used).
            class_name(str): A string to use for the class name in the decision rules
                ("class" if a string is not provided and the default name "class" should be used).
        """
        # could be done in one line pretty easily but this is easier to read
        ruleset = myutils.generate_decision_rules(self.tree, attribute_names, class_name)
        rules = "\n".join([ " ".join(["IF", *rule]) for rule in ruleset ])
        print(rules)
        # returns rules string after printing for convenience (e.g. for testing)
        return rules

    # BONUS METHOD
    def visualize_tree(self, dot_fname, pdf_fname, attribute_names=None):
        """BONUS: Visualizes a tree via the open source Graphviz graph visualization package and its DOT graph language (produces .dot and .pdf files).

        Args:
            dot_fname(str): The name of the .dot output file.
            pdf_fname(str): The name of the .pdf output file generated from the .dot file.
            attribute_names(list of str or None): A list of attribute names to use in the decision rules
                (None if a list is not provided and the default attribute names based on indexes (e.g. "att0", "att1", ...) should be used).

        Notes:
            Graphviz: https://graphviz.org/
            DOT language: https://graphviz.org/doc/info/lang.html
            You will need to install graphviz in the Docker container as shown in class to complete this method.
        """
        pass # TODO: (BONUS) fix this
