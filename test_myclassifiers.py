from mysklearn.myevaluation import train_test_split
import numpy as np
import scipy.stats as stats
from mysklearn import myutils
from mysklearn.mypytable import MyPyTable
from mysklearn.myclassifiers import MyRandomForestClassifier, MySimpleLinearRegressor, MyKNeighborsClassifier, MyNaiveBayesClassifier, DecisionTreeClassifier

# note: order is actual/received student value, expected/solution
def test_simple_linear_regressor_fit():
    linReg = MySimpleLinearRegressor()
    np.random.seed(0)
    x1 = [val for val in range(0, 100)]
    y1 = [value * 2 + np.random.normal(0, 25) for value in x1]
    linReg.fit([[x] for x in x1], y1)
    splr = stats.linregress(x1, y1)
    np.allclose([linReg.slope, linReg.intercept], [splr.slope, splr.intercept])

    x2 = [val for val in range(0, 100)]
    y2 = [value ** 3 - np.random.normal(0, 1000) for value in x2]
    linReg.fit([[x] for x in x2], y2)
    splr = stats.linregress(x2, y2)
    np.allclose([linReg.slope, linReg.intercept], [splr.slope, splr.intercept])

def test_simple_linear_regressor_predict():
    linReg = MySimpleLinearRegressor()
    np.random.seed(0)
    x1 = [val for val in range(100)]
    y1 = [value * 2 + np.random.normal(-1, 25) for value in x1]
    linReg.fit([[x] for x in x1], y1)
    splr = stats.linregress(x1, y1)

    x_test = [np.random.normal(-1000, 3000) for _ in range(25)]
    y_test = linReg.predict([[x] for x in x_test])
    y_actual = [ val * splr.slope + splr.intercept for val in x_test ]

    np.allclose(y_test, y_actual)

def test_kneighbors_classifier_kneighbors():
    kneigh = MyKNeighborsClassifier()

    # 4 instance
    X_train = [
        [7, 7],
        [7, 4],
        [3, 4],
        [1, 4]
    ]
    y_train = ["Bad", "Bad", "Good", "Good"]
    kneigh.fit(X_train, y_train)
    X_test = [
        [2, 5],
        [2, 7],
        [5, 4],
        [7, 4],
        [6, 4],
        [3, 6]
    ]
    y_expect = [
        [[0.372678, 0.372678, 0.897527], [2, 3, 1]],
        [[0.833333, 1.013794, 1.013794], [0, 2, 3]],
        [[0.333333, 0.333333, 0.666667], [2, 1, 3]],
        [[0.000000, 0.666667, 1.000000], [1, 2, 0]],
        [[0.166667, 0.500000, 0.833333], [1, 2, 3]],
        [[0.666667, 0.745356, 0.745356], [2, 3, 0]]
    ]
    y_test_distances, y_test_indecies = kneigh.kneighbors(X_test)
    assert len(y_expect) == len(y_test_distances)
    assert len(y_test_distances) == len(y_test_indecies)
    y_test = [y_test_distances, y_test_indecies]
    for i in range(len(y_expect)):
        np.allclose(y_expect[i][0], y_test[0][i])
        assert y_expect[i][1] == y_test[1][i]

    # 8 instance
    X_train = [
        [3, 2],
        [6, 6],
        [4, 1],
        [4, 4],
        [1, 2],
        [2, 0],
        [0, 3],
        [1, 6]
    ]
    y_train = ["no", "yes", "no", "no", "yes", "no", "yes", "yes"]
    kneigh.fit(X_train, y_train)
    X_test = [
        [2, 3],
        [5, 4],
        [1, 2],
        [0, 6],
        [3, 3],
        [2, 2]
    ]
    y_expect = [
        [[0.23570226039551584, 0.23570226039551587, 0.3333333333333333],  [4, 0, 6]],
        [[0.16666666666666674, 0.37267799624996495, 0.4714045207910317],  [3, 1, 0]],
        [[0.0, 0.23570226039551584, 0.33333333333333337],                 [4, 6, 0]],
        [[0.16666666666666666, 0.5, 0.6871842709362769],                  [7, 6, 4]],
        [[0.16666666666666669, 0.23570226039551578, 0.37267799624996495], [0, 3, 2]],
        [[0.16666666666666666, 0.16666666666666669, 0.3333333333333333],  [4, 0, 5]]
    ]
    y_test_distances, y_test_indecies = kneigh.kneighbors(X_test)
    assert len(y_expect) == len(y_test_distances)
    assert len(y_test_distances) == len(y_test_indecies)
    y_test = [y_test_distances, y_test_indecies]
    for i in range(len(y_expect)):
        np.allclose(y_expect[i][0], y_test[0][i])
        assert y_expect[i][1] == y_test[1][i]

    # Bramer
    kneigh = MyKNeighborsClassifier(5)
    X_train = [
        [0.8, 6.3],
        [1.4, 8.1],
        [2.1, 7.4],
        [2.6, 14.3],
        [6.8, 12.6],
        [8.8, 9.8],
        [9.2, 11.6],
        [10.8, 9.6],
        [11.8, 9.9],
        [12.4, 6.5],
        [12.8, 1.1],
        [14.0, 19.9],
        [14.2, 18.5],
        [15.6, 17.4],
        [15.8, 12.2],
        [16.6, 6.7],
        [17.4, 4.5],
        [18.2, 6.9],
        [19.0, 3.4],
        [19.6, 11.1]
    ]
    y_train = ["-", "-", "-", "+", "-", "+", "-", "+", "+", "+", "-", "-", "-", "-", "-", "+", "+", "+", "-", "+"]
    kneigh.fit(X_train, y_train)
    X_test = [[9.1, 11.0]]
    y_expect = [
        [[0.608, 1.237, 2.202, 2.802, 2.915], [6, 5, 7, 4, 8]]
    ]
    y_test_distances, y_test_indecies = kneigh.kneighbors(X_test)
    assert len(y_expect) == len(y_test_distances)
    assert len(y_test_distances) == len(y_test_indecies)
    y_test = [y_test_distances, y_test_indecies]
    for i in range(len(y_expect)):
        np.allclose(y_expect[i][0], y_test[0][i])
        assert y_expect[i][1] == y_test[1][i]

def test_kneighbors_classifier_predict():
    kneigh = MyKNeighborsClassifier()

     # 4 instance
    X_train = [
        [7, 7],
        [7, 4],
        [3, 4],
        [1, 4]
    ]
    y_train = ["Bad", "Bad", "Good", "Good"]
    kneigh.fit(X_train, y_train)
    X_test = [
        [2, 5],
        [2, 7],
        [5, 4],
        [7, 4],
        [6, 4],
        [3, 6]
    ]
    y_expect = ["Good", "Good", "Good", "Bad", "Good", "Good"]
    y_test = kneigh.predict(X_test)
    for i in range(len(y_expect)):
        assert y_expect[i] == y_test[i]

    # 8 instance
    X_train = [
        [3, 2],
        [6, 6],
        [4, 1],
        [4, 4],
        [1, 2],
        [2, 0],
        [0, 3],
        [1, 6]
    ]
    y_train = ["no", "yes", "no", "no", "yes", "no", "yes", "yes"]
    kneigh.fit(X_train, y_train)
    X_test = [
        [2, 3],
        [5, 4],
        [1, 2],
        [0, 6],
        [3, 3],
        [2, 2]
    ]
    y_expect = ["yes", "no", "yes", "yes", "no", "no"]
    y_test = kneigh.predict(X_test)
    for i in range(len(y_expect)):
        assert y_expect[i] == y_test[i]

    # Bramer
    kneigh = MyKNeighborsClassifier(5)
    X_train = [
        [0.8, 6.3],
        [1.4, 8.1],
        [2.1, 7.4],
        [2.6, 14.3],
        [6.8, 12.6],
        [8.8, 9.8],
        [9.2, 11.6],
        [10.8, 9.6],
        [11.8, 9.9],
        [12.4, 6.5],
        [12.8, 1.1],
        [14.0, 19.9],
        [14.2, 18.5],
        [15.6, 17.4],
        [15.8, 12.2],
        [16.6, 6.7],
        [17.4, 4.5],
        [18.2, 6.9],
        [19.0, 3.4],
        [19.6, 11.1]
    ]
    y_train = ["-", "-", "-", "+", "-", "+", "-", "+", "+", "+", "-", "-", "-", "-", "-", "+", "+", "+", "-", "+"]
    kneigh.fit(X_train, y_train)
    X_test = [[9.1, 11.0]]
    y_expect = ["+"]

    y_test = kneigh.predict(X_test)
    for i in range(len(y_expect)):
        assert y_expect[i] == y_test[i]

def compare_posteriors(expect, actual):
    for class_label in expect.keys():
        for col in range(len(expect[class_label])):
            for attr in expect[class_label][col].keys():
                if expect[class_label][col][attr] != actual[class_label][col][attr]:
                    return False
    return True

def compare_priors(expect, actual):
    for X in expect.keys():
        if expect[X] != actual [X]:
            return False
    return True

def test_naive_bayes_classifier_fit():
    nbayes = MyNaiveBayesClassifier()

    # Lab task 1
    X_train = [[1, 5],
               [2, 6],
               [1, 5],
               [1, 5],
               [1, 6],
               [2, 6],
               [1, 5],
               [1, 6]]
    y_train = ["yes", "yes", "no", "no", "yes", "no", "yes", "yes"]
    priors_expect = {
        "yes": 5.0/8.0,
        "no": 3.0/8.0
    }
    posteriors_expect = {
        'yes': [
            {
                1: 0.8,
                2: 0.2
            },
            {
                5: 0.4,
                6: 0.6
            }
        ],
        'no': [
            {
                1: (2.0/3.0),
                2: (1.0/3.0)
            },
            {
                5: (2.0/3.0),
                6: (1.0/3.0)
            }
        ]
    }

    nbayes.fit(X_train, y_train)
    assert compare_priors(priors_expect, nbayes.priors)
    assert compare_posteriors(posteriors_expect, nbayes.posteriors)


    # Lab task 2
    X_train = [["Rainy",    "Hot",  "High",   False],
               ["Rainy",    "Hot",  "High",   True ],
               ["Overcast", "Hot",  "High",   False],
               ["Sunny",    "Mild", "High",   False],
               ["Sunny",    "Cool", "Normal", False],
               ["Sunny",    "Cool", "Normal", True ],
               ["Overcast", "Cool", "Normal", True ],
               ["Rainy",    "Mild", "High",   False],
               ["Rainy",    "Cool", "Normal", False],
               ["Sunny",    "Mild", "Normal", False],
               ["Rainy",    "Mild", "Normal", True ],
               ["Overcast", "Mild", "High",   True ],
               ["Overcast", "Hot",  "Normal", False],
               ["Sunny",    "Mild", "High",   True ]]
    y_train = ["No", "No", "Yes", "Yes", "Yes", "No", "Yes", "No", "Yes", "Yes", "Yes", "Yes", "Yes", "No"]
    priors_expect = {
        "Yes": 9.0/14.0,
        "No": 5.0/14.0
    }
    posteriors_expect = {
        "Yes": [
            {
                "Rainy": 2.0/9.0,
                "Overcast": 4.0/9.0,
                "Sunny": 3.0/9.0
            },
            {
                "Hot": 2.0/9.0,
                "Mild": 4.0/9.0,
                "Cool": 3.0/9.0
            },
            {
                "High": 3.0/9.0,
                "Normal": 6.0/9.0
            },
            {
                False: 6.0/9.0,
                True: 3.0/9.0
            }
        ],
        "No": [
            {
                "Rainy": 3.0/5.0,
                "Overcast": 0.0/5.0,
                "Sunny": 2.0/5.0
            },
            {
                "Hot": 2.0/5.0,
                "Mild": 2.0/5.0,
                "Cool": 1.0/5.0
            },
            {
                "High": 4.0/5.0,
                "Normal": 1.0/5.0
            },
            {
                False: 2.0/5.0,
                True: 3.0/5.0
            }
        ]
    }

    nbayes.fit(X_train, y_train)
    assert compare_priors(priors_expect, nbayes.priors)
    assert compare_posteriors(posteriors_expect, nbayes.posteriors)


    # RQ5 (fake) iPhone purchases dataset
    iphone_col_names = ["standing", "job_status", "credit_rating", "buys_iphone"]
    iphone_table = [
        [1, 3, "fair", "no"],
        [1, 3, "excellent", "no"],
        [2, 3, "fair", "yes"],
        [2, 2, "fair", "yes"],
        [2, 1, "fair", "yes"],
        [2, 1, "excellent", "no"],
        [2, 1, "excellent", "yes"],
        [1, 2, "fair", "no"],
        [1, 1, "fair", "yes"],
        [2, 2, "fair", "yes"],
        [1, 2, "excellent", "yes"],
        [2, 2, "excellent", "yes"],
        [2, 3, "fair", "yes"],
        [2, 2, "excellent", "no"],
        [2, 3, "fair", "yes"]
    ]
    data_in = MyPyTable(iphone_col_names, iphone_table)
    X_train = myutils.prepare_x_list_from_mypytable(data_in, ["standing", "job_status", "credit_rating"])
    y_train = data_in.get_column("buys_iphone")
    priors_expect = {
        "yes": 10/15,
        "no": 5/15
    }
    posteriors_expect = {
        "yes": [
            {
                1: 2/10,
                2: 8/10
            },
            {
                1: 3/10,
                2: 4/10,
                3: 3/10
            },
            {
                "fair": 7/10,
                "excellent": 3/10
            }
        ],
        "no": [
            {
                1: 3/5,
                2: 2/5
            },
            {
                1: 1/5,
                2: 2/5,
                3: 2/5
            },
            {
                "fair": 2/5,
                "excellent": 3/5
            }
        ]
    }

    nbayes.fit(X_train, y_train)
    assert compare_priors(priors_expect, nbayes.priors)
    assert compare_posteriors(posteriors_expect, nbayes.posteriors)


    # Bramer 3.2 train dataset
    train_col_names = ["day", "season", "wind", "rain", "class"]
    train_table = [
        ["weekday", "spring", "none", "none", "on time"],
        ["weekday", "winter", "none", "slight", "on time"],
        ["weekday", "winter", "none", "slight", "on time"],
        ["weekday", "winter", "high", "heavy", "late"],
        ["saturday", "summer", "normal", "none", "on time"],
        ["weekday", "autumn", "normal", "none", "very late"],
        ["holiday", "summer", "high", "slight", "on time"],
        ["sunday", "summer", "normal", "none", "on time"],
        ["weekday", "winter", "high", "heavy", "very late"],
        ["weekday", "summer", "none", "slight", "on time"],
        ["saturday", "spring", "high", "heavy", "cancelled"],
        ["weekday", "summer", "high", "slight", "on time"],
        ["saturday", "winter", "normal", "none", "late"],
        ["weekday", "summer", "high", "none", "on time"],
        ["weekday", "winter", "normal", "heavy", "very late"],
        ["saturday", "autumn", "high", "slight", "on time"],
        ["weekday", "autumn", "none", "heavy", "on time"],
        ["holiday", "spring", "normal", "slight", "on time"],
        ["weekday", "spring", "normal", "none", "on time"],
        ["weekday", "spring", "normal", "slight", "on time"]
    ]
    data_in = MyPyTable(train_col_names, train_table)
    X_train = myutils.prepare_x_list_from_mypytable(data_in, train_col_names[:-1])
    y_train = data_in.get_column(train_col_names[-1])
    priors_expect = {
        "on time": 14/20,
        "late": 2/20,
        "very late": 3/20,
        "cancelled": 1/20
    }
    posteriors_expect = {
        "on time": [
            {
                "weekday": 9/14,
                "saturday": 2/14,
                "holiday": 2/14,
                "sunday": 1/14
            },
            {
                "spring": 4/14,
                "winter": 2/14,
                "summer": 6/14,
                "autumn": 2/14
            },
            {
                "none": 5/14,
                "high": 4/14,
                "normal": 5/14
            },
            {
                "none": 5/14,
                "slight": 8/14,
                "heavy": 1/14
            },
        ],
        "late": [
            {
                "weekday": 1/2,
                "saturday": 1/2,
                "holiday": 0/2,
                "sunday": 0/2
            },
            {
                "spring": 0/2,
                "winter": 2/2,
                "summer": 0/2,
                "autumn": 0/2
            },
            {
                "none": 0/2,
                "high": 1/2,
                "normal": 1/2
            },
            {
                "none": 1/2,
                "slight": 0/2,
                "heavy": 1/2
            },
        ],
        "very late": [
            {
                "weekday": 3/3,
                "saturday": 0/3,
                "holiday": 0/3,
                "sunday": 0/3
            },
            {
                "spring": 0/3,
                "winter": 2/3,
                "summer": 0/3,
                "autumn": 1/3
            },
            {
                "none": 0/3,
                "high": 1/3,
                "normal": 2/3
            },
            {
                "none": 1/3,
                "slight": 0/3,
                "heavy": 2/3
            },
        ],
        "cancelled": [
            {
                "weekday": 0/1,
                "saturday": 1/1,
                "holiday": 0/1,
                "sunday": 0/1
            },
            {
                "spring": 1/1,
                "winter": 0/1,
                "summer": 0/1,
                "autumn": 0/1
            },
            {
                "none": 0/1,
                "high": 1/1,
                "normal": 0/1
            },
            {
                "none": 0/1,
                "slight": 0/1,
                "heavy": 1/1
            },
        ]
    }

    nbayes.fit(X_train, y_train)
    assert compare_priors(priors_expect, nbayes.priors)
    assert compare_posteriors(posteriors_expect, nbayes.posteriors)

def test_naive_bayes_classifier_predict():
    nbayes = MyNaiveBayesClassifier()

    # Lab task 1
    X_train = [[1, 5],
               [2, 6],
               [1, 5],
               [1, 5],
               [1, 6],
               [2, 6],
               [1, 5],
               [1, 6]]
    y_train = ["yes", "yes", "no", "no", "yes", "no", "yes", "yes"]
    X_list = [[1,6],[2,5],[1,6],[1,5],[2,6]]
    predictions_expect = ["yes", "no", "yes", "yes", "yes"]

    nbayes.fit(X_train, y_train)
    predictions_actual = nbayes.predict(X_list)
    assert myutils.compare_str_lists(predictions_expect, predictions_actual)


    # Lab task 2
    """
    posteriors_expect = {
        ("Rainy", "Hot", "High", True): "No",
        ("Rainy", "Hot", "High", False): "No",
        ("Rainy", "Hot", "Normal", True): "No",
        ("Rainy", "Hot", "Normal", False): "Yes",
        ("Rainy", "Mild", "High", True): "No",
        ("Rainy", "Mild", "High", False): "No",
        ("Rainy", "Mild", "Normal", True): "Yes",
        ("Rainy", "Mild", "Normal", False): "Yes",
        ("Rainy", "Cool", "High", True): "No",
        ("Rainy", "Cool", "High", False): "No",
        ("Rainy", "Cool", "Normal", True): "Yes",
        ("Rainy", "Cool", "Normal", False): "Yes",
        ("Overcast", "Hot", "High", True): "Yes",
        ("Overcast", "Hot", "High", False): "Yes",
        ("Overcast", "Hot", "Normal", True): "Yes",
        ("Overcast", "Hot", "Normal", False): "Yes",
        ("Overcast", "Mild", "High", True): "Yes",
        ("Overcast", "Mild", "High", False): "Yes",
        ("Overcast", "Mild", "Normal", True): "Yes",
        ("Overcast", "Mild", "Normal", False): "Yes",
        ("Overcast", "Cool", "High", True): "Yes",
        ("Overcast", "Cool", "High", False): "Yes",
        ("Overcast", "Cool", "Normal", True): "Yes",
        ("Overcast", "Cool", "Normal", False): "Yes",
        ("Sunny", "Hot", "High", True): "No",
        ("Sunny", "Hot", "High", False): "No",
        ("Sunny", "Hot", "Normal", True): "Yes",
        ("Sunny", "Hot", "Normal", False): "Yes",
        ("Sunny", "Mild", "High", True): "No",
        ("Sunny", "Mild", "High", False): "Yes",
        ("Sunny", "Mild", "Normal", True): "Yes",
        ("Sunny", "Mild", "Normal", False): "Yes",
        ("Sunny", "Cool", "High", True): "No",
        ("Sunny", "Cool", "High", False): "Yes",
        ("Sunny", "Cool", "Normal", True): "Yes",
        ("Sunny", "Cool", "Normal", False): "Yes"
    }
    """


    # RQ5 (fake) iPhone purchases dataset
    iphone_col_names = ["standing", "job_status", "credit_rating", "buys_iphone"]
    iphone_table = [
        [1, 3, "fair", "no"],
        [1, 3, "excellent", "no"],
        [2, 3, "fair", "yes"],
        [2, 2, "fair", "yes"],
        [2, 1, "fair", "yes"],
        [2, 1, "excellent", "no"],
        [2, 1, "excellent", "yes"],
        [1, 2, "fair", "no"],
        [1, 1, "fair", "yes"],
        [2, 2, "fair", "yes"],
        [1, 2, "excellent", "yes"],
        [2, 2, "excellent", "yes"],
        [2, 3, "fair", "yes"],
        [2, 2, "excellent", "no"],
        [2, 3, "fair", "yes"]
    ]
    data_in = MyPyTable(iphone_col_names, iphone_table)
    X_train = myutils.prepare_x_list_from_mypytable(data_in, ["standing", "job_status", "credit_rating"])
    y_train = data_in.get_column("buys_iphone")
    X_list = [[2, 2, "fair"], [1, 1, "excellent"]]
    predictions_expect = ["yes", "no"]

    nbayes.fit(X_train, y_train)
    predictions_actual = nbayes.predict(X_list)
    assert myutils.compare_str_lists(predictions_expect, predictions_actual)


    # Bramer 3.2 train dataset
    train_col_names = ["day", "season", "wind", "rain", "class"]
    train_table = [
        ["weekday", "spring", "none", "none", "on time"],
        ["weekday", "winter", "none", "slight", "on time"],
        ["weekday", "winter", "none", "slight", "on time"],
        ["weekday", "winter", "high", "heavy", "late"],
        ["saturday", "summer", "normal", "none", "on time"],
        ["weekday", "autumn", "normal", "none", "very late"],
        ["holiday", "summer", "high", "slight", "on time"],
        ["sunday", "summer", "normal", "none", "on time"],
        ["weekday", "winter", "high", "heavy", "very late"],
        ["weekday", "summer", "none", "slight", "on time"],
        ["saturday", "spring", "high", "heavy", "cancelled"],
        ["weekday", "summer", "high", "slight", "on time"],
        ["saturday", "winter", "normal", "none", "late"],
        ["weekday", "summer", "high", "none", "on time"],
        ["weekday", "winter", "normal", "heavy", "very late"],
        ["saturday", "autumn", "high", "slight", "on time"],
        ["weekday", "autumn", "none", "heavy", "on time"],
        ["holiday", "spring", "normal", "slight", "on time"],
        ["weekday", "spring", "normal", "none", "on time"],
        ["weekday", "spring", "normal", "slight", "on time"]
    ]
    data_in = MyPyTable(train_col_names, train_table)
    X_train = myutils.prepare_x_list_from_mypytable(data_in, train_col_names[:-1])
    y_train = data_in.get_column(train_col_names[-1])
    X_list = [["weekday", "winter", "high", "heavy"], ["weekday", "summer", "high", "heavy"], ["sunday", "summer", "normal", "slight"]]
    predictions_expect = ["very late", "on time", "on time"]

    nbayes.fit(X_train, y_train)
    predictions_actual = nbayes.predict(X_list)
    assert myutils.compare_str_lists(predictions_expect, predictions_actual)

def test_decision_tree_classifier_fit():
    f = MyRandomForestClassifier(F=0.5, M=10, max_threads=1)

    # interview dataset
    interview_header = ["level", "lang", "tweets", "phd", "interviewed_well"]
    X_train = [
        ["Senior", "Java", "no", "no"],
        ["Senior", "Java", "no", "yes"],
        ["Mid", "Python", "no", "no"],
        ["Junior", "Python", "no", "no"],
        ["Junior", "R", "yes", "no"],
        ["Junior", "R", "yes", "yes"],
        ["Mid", "R", "yes", "yes"],
        ["Senior", "Python", "no", "no"],
        ["Senior", "R", "yes", "no"],
        ["Junior", "Python", "yes", "no"],
        ["Senior", "Python", "yes", "yes"],
        ["Mid", "Python", "no", "yes"],
        ["Mid", "Java", "yes", "no"],
        ["Junior", "Python", "no", "yes"]
    ]
    y_train = ["False", "False", "True", "True", "True", "False", "True", "False", "True", "True", "True", "True", "True", "False"]

    f.fit(X_train, y_train)
    print(len(f.forest))

    # bramer degrees dataset
    degrees_header = ["SoftEng", "ARIN", "HCI", "CSA", "Project", "Class"]
    X_train = [
        ["A", "B", "A", "B", "B"],
        ["A", "B", "B", "B", "A"],
        ["A", "A", "A", "B", "B"],
        ["B", "A", "A", "B", "B"],
        ["A", "A", "B", "B", "A"],
        ["B", "A", "A", "B", "B"],
        ["A", "B", "B", "B", "B"],
        ["A", "B", "B", "B", "B"],
        ["A", "A", "A", "A", "A"],
        ["B", "A", "A", "B", "B"],
        ["B", "A", "A", "B", "B"],
        ["A", "B", "B", "A", "B"],
        ["B", "B", "B", "B", "A"],
        ["A", "A", "B", "A", "B"],
        ["B", "B", "B", "B", "A"],
        ["A", "A", "B", "B", "B"],
        ["B", "B", "B", "B", "B"],
        ["A", "A", "B", "A", "A"],
        ["B", "B", "B", "A", "A"],
        ["B", "B", "A", "A", "B"],
        ["B", "B", "B", "B", "A"],
        ["B", "A", "B", "A", "B"],
        ["A", "B", "B", "B", "A"],
        ["A", "B", "A", "B", "B"],
        ["B", "A", "B", "B", "B"],
        ["A", "B", "B", "B", "B"],
    ]
    y_train = ["SECOND", "FIRST", "SECOND", "SECOND", "FIRST", "SECOND", "SECOND", "SECOND", "FIRST", "SECOND", "SECOND", "SECOND", "SECOND", "FIRST", "SECOND", "SECOND", "SECOND", "FIRST", "SECOND", "SECOND", "SECOND", "SECOND", "FIRST", "SECOND", "SECOND", "SECOND"]

    f.fit(X_train, y_train)
    print(len(f.forest))

def test_random_forest_classifier_fit_big():
    f = MyRandomForestClassifier(F=0.33333, M=100, max_threads=6)

    data = MyPyTable().load_from_file("input_data/titanic.txt")
    header = data.column_names[:-1]
    X_train = myutils.prepare_x_list_from_mypytable(data, header)
    y_train = data.get_column(-1)

    f.fit(X_train, y_train)

def test_random_forest_classifier_predict_big():
    f = MyRandomForestClassifier(F=0.5, num_trees=32, M=3, max_threads=6)

    data = MyPyTable().load_from_file("input_data/titanic.txt")
    header = data.column_names[:-1]
    X = myutils.prepare_x_list_from_mypytable(data, header)
    y = data.get_column(-1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=100)

    f.fit(X_train, y_train)
    y_actual = f.predict(X_test)
    assert myutils.compare_str_lists_margin(y_test, y_actual, margin=0.7)


def test_decision_tree_classifier_predict():
    f = MyRandomForestClassifier(F=0.333, M=10, max_threads=1)

    # interview dataset
    interview_header = ["level", "lang", "tweets", "phd", "interviewed_well"]
    X_train = [
        ["Senior", "Java", "no", "no"],
        ["Senior", "Java", "no", "yes"],
        ["Mid", "Python", "no", "no"],
        ["Junior", "Python", "no", "no"],
        ["Junior", "R", "yes", "no"],
        ["Junior", "R", "yes", "yes"],
        ["Mid", "R", "yes", "yes"],
        ["Senior", "Python", "no", "no"],
        ["Senior", "R", "yes", "no"],
        ["Junior", "Python", "yes", "no"],
        ["Senior", "Python", "yes", "yes"],
        ["Mid", "Python", "no", "yes"],
        ["Mid", "Java", "yes", "no"],
        ["Junior", "Python", "no", "yes"]
    ]
    y_train = ["False", "False", "True", "True", "True", "False", "True", "False", "True", "True", "True", "True", "True", "False"]
    X_test = [["Junior", "Java", "yes", "no"], ["Junior", "Java", "yes", "yes"]]
    predict_expect = ["True", "False"]

    f.fit(X_train, y_train)
    predict_actual = f.predict(X_test)
    print(predict_actual)
    assert myutils.compare_str_lists(predict_expect, predict_actual)


    # bramer degrees dataset
    degrees_header = ["SoftEng", "ARIN", "HCI", "CSA", "Project", "Class"]
    X_train = [
        ["A", "B", "A", "B", "B"],
        ["A", "B", "B", "B", "A"],
        ["A", "A", "A", "B", "B"],
        ["B", "A", "A", "B", "B"],
        ["A", "A", "B", "B", "A"],
        ["B", "A", "A", "B", "B"],
        ["A", "B", "B", "B", "B"],
        ["A", "B", "B", "B", "B"],
        ["A", "A", "A", "A", "A"],
        ["B", "A", "A", "B", "B"],
        ["B", "A", "A", "B", "B"],
        ["A", "B", "B", "A", "B"],
        ["B", "B", "B", "B", "A"],
        ["A", "A", "B", "A", "B"],
        ["B", "B", "B", "B", "A"],
        ["A", "A", "B", "B", "B"],
        ["B", "B", "B", "B", "B"],
        ["A", "A", "B", "A", "A"],
        ["B", "B", "B", "A", "A"],
        ["B", "B", "A", "A", "B"],
        ["B", "B", "B", "B", "A"],
        ["B", "A", "B", "A", "B"],
        ["A", "B", "B", "B", "A"],
        ["A", "B", "A", "B", "B"],
        ["B", "A", "B", "B", "B"],
        ["A", "B", "B", "B", "B"],
    ]
    y_train = ["SECOND", "FIRST", "SECOND", "SECOND", "FIRST", "SECOND", "SECOND", "SECOND", "FIRST", "SECOND", "SECOND", "SECOND", "SECOND", "FIRST", "SECOND", "SECOND", "SECOND", "FIRST", "SECOND", "SECOND", "SECOND", "SECOND", "FIRST", "SECOND", "SECOND", "SECOND"]
    X_test = [["B", "B", "B", "B", "B"], ["A", "A", "A", "A", "A"], ["A", "A", "A", "A", "B"]]
    predict_expect = ["SECOND", "FIRST", "FIRST"]

    f.fit(X_train, y_train)
    predict_actual = f.predict(X_test)
    assert myutils.compare_str_lists(predict_expect, predict_actual)


def test_decision_tree_classifier_print_rules():
    dtree = DecisionTreeClassifier()

    # interview dataset
    header = ["level", "lang", "tweets", "phd", "interviewed_well"]
    X_train = [
        ["Senior", "Java", "no", "no"],
        ["Senior", "Java", "no", "yes"],
        ["Mid", "Python", "no", "no"],
        ["Junior", "Python", "no", "no"],
        ["Junior", "R", "yes", "no"],
        ["Junior", "R", "yes", "yes"],
        ["Mid", "R", "yes", "yes"],
        ["Senior", "Python", "no", "no"],
        ["Senior", "R", "yes", "no"],
        ["Junior", "Python", "yes", "no"],
        ["Senior", "Python", "yes", "yes"],
        ["Mid", "Python", "no", "yes"],
        ["Mid", "Java", "yes", "no"],
        ["Junior", "Python", "no", "yes"]
    ]
    y_train = ["False", "False", "True", "True", "True", "False", "True", "False", "True", "True", "True", "True", "True", "False"]
    rules_expect =   "\n".join(["IF level == Senior AND tweets == yes THEN interviewed_well = True",
                                "IF level == Senior AND tweets == no THEN interviewed_well = False",
                                "IF level == Mid THEN interviewed_well = True",
                                "IF level == Junior AND phd == no THEN interviewed_well = True",
                                "IF level == Junior AND phd == yes THEN interviewed_well = False"])

    dtree.fit(X_train, y_train)
    rules_actual = dtree.print_decision_rules(header[:-1], header[-1])
    r1 = rules_expect.split("\n")
    r2 = rules_actual.split("\n")
    r1.sort()
    r2.sort()
    assert myutils.compare_str_lists(r1, r2)


    # bramer degrees dataset
    header = ["SoftEng", "ARIN", "HCI", "CSA", "Project", "Class"]
    X_train = [
        ["A", "B", "A", "B", "B"],
        ["A", "B", "B", "B", "A"],
        ["A", "A", "A", "B", "B"],
        ["B", "A", "A", "B", "B"],
        ["A", "A", "B", "B", "A"],
        ["B", "A", "A", "B", "B"],
        ["A", "B", "B", "B", "B"],
        ["A", "B", "B", "B", "B"],
        ["A", "A", "A", "A", "A"],
        ["B", "A", "A", "B", "B"],
        ["B", "A", "A", "B", "B"],
        ["A", "B", "B", "A", "B"],
        ["B", "B", "B", "B", "A"],
        ["A", "A", "B", "A", "B"],
        ["B", "B", "B", "B", "A"],
        ["A", "A", "B", "B", "B"],
        ["B", "B", "B", "B", "B"],
        ["A", "A", "B", "A", "A"],
        ["B", "B", "B", "A", "A"],
        ["B", "B", "A", "A", "B"],
        ["B", "B", "B", "B", "A"],
        ["B", "A", "B", "A", "B"],
        ["A", "B", "B", "B", "A"],
        ["A", "B", "A", "B", "B"],
        ["B", "A", "B", "B", "B"],
        ["A", "B", "B", "B", "B"],
    ]
    y_train = ["SECOND", "FIRST", "SECOND", "SECOND", "FIRST", "SECOND", "SECOND", "SECOND", "FIRST", "SECOND", "SECOND", "SECOND", "SECOND", "FIRST", "SECOND", "SECOND", "SECOND", "FIRST", "SECOND", "SECOND", "SECOND", "SECOND", "FIRST", "SECOND", "SECOND", "SECOND"]
    rules_expect = "IF SoftEng == A AND Project == A THEN Class = FIRST\nIF SoftEng == A AND Project == B AND CSA == A AND ARIN == A THEN Class = FIRST\nIF SoftEng == A AND Project == B AND CSA == A AND ARIN == B THEN Class = SECOND\nIF SoftEng == A AND Project == B AND CSA == B THEN Class = SECOND\nIF SoftEng == B THEN Class = SECOND"

    dtree.fit(X_train, y_train)
    rules_actual = dtree.print_decision_rules(header[:-1], header[-1])
    r1 = rules_expect.split("\n")
    r2 = rules_actual.split("\n")
    r1.sort()
    r2.sort()
    assert myutils.compare_str_lists(r1, r2)
