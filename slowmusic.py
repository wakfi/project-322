# some useful mysklearn package import statements and reloads
import importlib, copy

import mysklearn.myutils
importlib.reload(mysklearn.myutils)
import mysklearn.myutils as myutils

# uncomment once you paste your mypytable.py into mysklearn package
import mysklearn.mypytable
importlib.reload(mysklearn.mypytable)
from mysklearn.mypytable import MyPyTable

import mysklearn.myclassifiers
importlib.reload(mysklearn.myclassifiers)
from mysklearn.myclassifiers import MyKNeighborsClassifier, MySimpleLinearRegressor, MyNaiveBayesClassifier, DecisionTreeClassifier, MyRandomForestClassifier

import mysklearn.myevaluation
importlib.reload(mysklearn.myevaluation)
import mysklearn.myevaluation as myevaluation

print(1)
data = MyPyTable().load_from_file("input_data/tracks_features.csv")
print(2)
data.convert_to_numeric()
print(3)
raw_data = copy.deepcopy(data.data)
print(4)

invert_col_names = True
x_col_names = ["album_id", "artist_ids", "id", "release_date", "year", "name"]
y_col_name = "year"
if invert_col_names:
    x_col_names = list(set(data.column_names).difference(set(x_col_names)))
    print(x_col_names)

data.apply_to_column("duration_ms", lambda x: [ y // 1000 for y in x ])

continuous_cols = ["danceability","energy","key","loudness","mode","speechiness","acousticness","instrumentalness","liveness","valence","tempo","duration_ms"]
myutils.bucketize(data.get_column(continuous_cols[0]))
for col_name in continuous_cols:
        data.apply_to_column(col_name, myutils.bucketize)

x_data = myutils.prepare_x_list_from_mypytable(data, x_col_names)
y_data = data.get_column(y_col_name)

data.pretty_print()
