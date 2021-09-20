from mysklearn import myutils
from mysklearn.myclassifiers import MyRandomForestClassifier
from mysklearn.mypytable import MyPyTable
from datetime import datetime

if __name__ == "__main__":
    f = MyRandomForestClassifier(F=0.3333, M=1000, max_threads=6)

    data = MyPyTable().load_from_file("input_data/auto-data-removed-NA.txt")
    data.convert_to_numeric()
    header = data.column_names[1:data.column_names.index('car name')] + data.column_names[data.column_names.index('car name')+1:]
    X_train = myutils.prepare_x_list_from_mypytable(data, header)
    y_train = myutils.mpg_list_to_rank_list(data.get_column(0))

    j1 = header.index("weight")
    j2 = header.index("msrp")
    j3 = header.index("displacement")
    j4 = header.index("horsepower")
    j5 = header.index("acceleration")
    for i in range(len(X_train)):
        X_train[i][j1] = myutils.weight_to_categorical(X_train[i][j1])
        X_train[i][j2] = X_train[i][j2] // 1000
        X_train[i][j3] = X_train[i][j3] // 500
        X_train[i][j4] = X_train[i][j4] // 20
        X_train[i][j5] = X_train[i][j5] // 2

    s = datetime.now()
    f.fit(X_train, y_train)
    e = datetime.now()
    print("time:", e-s)
    # [print(t.tree, end='\n\n') for t in f.forest]
