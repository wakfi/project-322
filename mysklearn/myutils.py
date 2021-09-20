from typing import Callable, Iterable, Literal, TypeVar, Union
from numpy.core.numeric import Infinity
from mysklearn.mypytable import MyPyTable
import random
import math

def avg(n: list)-> float:
    return sum(n) * 1.0 / len(n)

def distance(a: list, b: list)-> float:
    if is_numeric_list_strict(a) and is_numeric_list_strict(b):
        return math.sqrt(sum((a[i] - b[i])**2 for i in range(len(a))))
    else:
        return math.sqrt(sum(0 if a[i] == b[i] else 1 for i in range(len(a))))

def is_numeric(val)-> bool:
    try:
        float(val)
    except:
        return False
    return True

def is_numeric_list(values: list)-> bool:
    for val in values:
        if not is_numeric(val):
            return False
    return True

def is_numeric_strict(val)-> bool:
    return isinstance(val, float) or isinstance(val, int)

def is_numeric_list_strict(values: list)-> bool:
    for val in values:
        if not is_numeric_strict(val):
            return False
    return True

def normalize_value(value, min_val, range_val)-> float:
    return (value - min_val) / (range_val * 1.0)

def normalize(values: list, *, min_val=None, range_val=None)-> list:
    if min_val is None:
        min_val = min(values)
    if range_val is None:
        range_val = (max(values) - min_val) * 1.0
    return [(value - min_val) / range_val for value in values]

def normalize_table(table: list, *, against: list = None)-> list:
    height = len(table)
    table_copy = [[] for _ in range(height)]
    min_val = None
    range_val = None
    for j in range(len(table[0])):
        column = get_column(table, j)
        if is_numeric_list_strict(column):
            if against is not None:
                against_column = get_column(against, j)
                min_val = min(against_column)
                range_val = (max(against_column) - min_val) * 1.0
            column = normalize(get_column(table, j), min_val=min_val, range_val=range_val)
        for i in range(height):
            table_copy[i].append(column[i])
    return table_copy

def get_column(table: list, col: int)-> list:
    return [table[i][col] for i in range(len(table))]

def gen_shuffle_indecies(size: int) -> list:
    indecies = [ i for i in range(size) ]
    for i in range(size - 1):
        j = random.randrange(i, size)
        indecies[i], indecies[j] = indecies[j], indecies[i]
    return indecies

def shuffle(l: list, seed=None)-> list:
    if seed is not None:
        random.seed(seed)
    if len(l) == 0:
        return l
    idx = gen_shuffle_indecies(len(l))
    return [l[i] for i in idx]

def reorder(new_indecies: list, *args):
    """Reorder one or more lists using the provided order of indecies. With multiple lists, all will be reorderd in parallel"""
    return tuple([[arg[i] for i in new_indecies] for arg in args])

def shuffle_in_parallel(*args):
    size = len(args)
    if size == 0:
        return ()
    new_indecies = gen_shuffle_indecies(size)
    return reorder(new_indecies, *args)

def sort_in_parallel(*args, sortFunc=None):
    if len(args) == 0:
        return ()
    sort_arg = [(args[0][j], j) for j in range(len(args[0]))]
    sort_arg.sort(key=sortFunc)
    new_indecies = [index for _, index in sort_arg]
    return reorder(new_indecies, *args)

def stratify(y: list)-> list:
    class_labels = []
    bins = []
    for label in y:
        if class_labels.count(label) == 0:
            class_labels.append(label)
            bins.append([])
    for i in range(len(y)):
        index = class_labels.index(y[i])
        bins[index].append(i)
    return bins, class_labels

def mpg_list_to_rank_list(mpg_list: "list[float]")-> "list[int]":
    return [ mpg_to_rank(mpg) for mpg in mpg_list ]

def mpg_to_rank(mpg: float)-> int:
    if mpg <= 13:
        return 1
    elif mpg < 15:
        return 2
    elif mpg < 17:
        return 3
    elif mpg < 20:
        return 4
    elif mpg < 24:
        return 5
    elif mpg < 27:
        return 6
    elif mpg < 31:
        return 7
    elif mpg < 37:
        return 8
    elif mpg < 45:
        return 9
    else:
        return 10

def weight_to_categorical(weight):
    if weight < 2000:
        return 1
    elif weight < 2500:
        return 2
    elif weight < 3000:
        return 3
    elif weight < 3500:
        return 4
    else:
        return 5

def prepare_x_list_from_mypytable(table: MyPyTable, col_names: "list[str]")-> "list[list]":
    return distribute([ [ x for x in table.get_column(col_name) ] for col_name in col_names ])

def get_instances_for_x_list_from_mypytable(table: MyPyTable, X_test: "list[list]", X_col_names: "list[str]", y_test:list=None, y_col_name:str=None, *, stringify:bool=False)-> "list[list]":
    all_col_names = [x for x in X_col_names]
    all_test = [ [x for x in inst] for inst in X_test ]
    if y_test is not None and y_col_name is not None:
        all_col_names.append(y_col_name)
        [ all_test[i].append(y_test[i]) for i in range(len(y_test)) ]
    all_inst_list = distribute([ table.get_column(name) for name in all_col_names ])
    inst_list = []
    test_len = len(all_test[0])
    for test_inst in all_test:
        for i in range(len(all_inst_list)):
            j = 0
            col = all_inst_list[i]
            for j in range(test_len):
                if col[j] != test_inst[j]:
                    break
            if j == test_len-1:
                inst_list.append(table.data[i])
                break
    if stringify:
        inst_list = [ [ str(val) for val in inst ] for inst in inst_list ]
    return inst_list

def distribute(y: list):
    """Like stratify but more general. Closer to the "dealing cards" visual"""
    length = len(y)
    if(length == 0):
        return []
    width = len(y[0])
    return [ [ y[j][i] for j in range(length) ] for i in range(width) ]

def compute_confusion_matrix_accuracy(confusion_matrix: "list[list[int]]")-> float:
    size = len(confusion_matrix)
    if size == 0:
        return 1.0
    R = sum_matrix(confusion_matrix)
    return avg([ (R - sum([confusion_matrix[i][j] + confusion_matrix[j][i] for j in range(size) if j != i])) / R for i in range(size) ])

def sum_matrix(matrix: "list[list]"):
    return sum([ sum(row) for row in matrix ])

def dict_max_val(dic: dict):
    if(len(dic) == 0):
        return None
    value = -Infinity
    for v in dic.values():
        if v > value:
            value = v
    return value

def dict_max_key(dic: dict):
    if(len(dic) == 0):
        return None
    key = None
    value = -Infinity
    for k, v in dic.items():
        if v > value:
            key = k
            value = v
    return key

def dict_min_key(dic: dict):
    if(len(dic) == 0):
        return None
    key = None
    value = Infinity
    for k, v in dic.items():
        if v < value:
            key = k
            value = v
    return key

def dict_pick_key(dic: dict, comparator: Callable, start=None):
    if(len(dic) == 0):
        return None
    key = None
    value = start
    if value is None:
        value = [*dic.values()][0]
    for k, v in dic.items():
        if comparator(value, v) > 0:
            key = k
            value = v
    return key

def compare_str_lists(l1, l2):
    if len(l1) != len(l2):
        return False
    for i in range(len(l1)):
        if l1[i] != l2[i]:
            return False
    return True

def compare_str_lists_margin(l1, l2, margin=0.83):
    if len(l1) < len(l2):
        for _ in range(len(l2) - len(l1)):
            l1.append(None)
    elif len(l1) > len(l2):
        for _ in range(len(l1) - len(l2)):
            l2.append(None)
    # print(avg([1 if l1[i] != None and l1[i] == l2[i] else 0 for i in range(len(l1))]))
    return avg([1 if l1[i] != None and l1[i] == l2[i] else 0 for i in range(len(l1))]) >= margin

def compare_tree_lists(t1, t2):
    if len(t1) != len(t2):
        return False
    if t1[0] == "Leaf":
        for i in range(len(t1)):
            if t1[i] != t2[i]:
                return False
        return True
    # assert t1[0] == t2[0] == "Attribute"
    if t1[1] != t2[1]:
        return False
    for n1 in t1[2:]:
        # assert n1[0] == "Value"
        n2 = None
        for node in t2[2:]:
            if node[1] == n1[1]:
                n2 = node
                break
        if n2 is None:
            return False
        # assert n2[0] == "Value"
        if not compare_tree_lists(n1[2], n2[2]):
            return False
    return True

def generate_decision_rules(tree: list, attribute_names:"list[str]" = None, class_name:str = "class")-> str:
    clauses = []
    if(tree[0] == "Leaf"):
        return [[generate_decision_rule_fragment(tree, attribute_names, class_name)]]
    for node in tree[2:]:
        subrules = generate_decision_rules(node[2], attribute_names, class_name)
        base_rule = " ".join([generate_decision_rule_fragment(tree, attribute_names, class_name), "==", generate_decision_rule_fragment(node, attribute_names, class_name)])
        for subrule in subrules:
            clauses.append([" AND ".join([base_rule, *subrule[:-1]]), subrule[-1]])
    return clauses

def generate_decision_rule_fragment(node: list, attribute_names:"list[str]" = None, class_name:str = "class")-> str:
    if len(node) == 0:
        return ""
    if node[0] == "Leaf":
        return f"THEN {class_name} = {node[1]}"
    if node[0] == "Value":
        return f"{node[1]}"
    if node[0] == "Attribute":
        return f"{attribute_names[node[1]]}" if attribute_names is not None else f"att{node[1]}"
    return ""

def multi_iter(*iters: Iterable):
    if len(iters) == 0:
        return
    iters = [ iter(it) for it in iters ]
    try:
        while True:
            try:
                yield (next(iter) for iter in iters)
            except:
                return
    except:
        pass

R = TypeVar("R")
def bind(func: Callable[..., R], *func_args, **func_kwargs)-> Callable[[], R]:
    def bound_func()-> R:
        return func(*func_args, **func_kwargs)
    return bound_func

def entropy(*p: "Union[float, int]", total:int = None)-> float:
    if total is not None:
        total = float(total)
        p = [ val / total for val in p ]
    return -sum([ p[i]*math.log2(p[i]) for i in range(len(p)) ])

def interestingness(nleft: int, nright: int, nboth: int, ntotal: int)-> "dict[str, float]":
    measure = {}
    measure["confidence"] = float(nboth / nleft)
    measure["support"] = float(nboth / ntotal)
    measure["completeness"] = float(nboth / nright)
    return measure

def itemsets(transactions: "list[set]", minsup: float = 0.25):
    items = { item for transaction in transactions for item in transaction }
    ck = sorted(items)
    lk = [ {item} for item in ck ]
    leni = len(lk[0]) - 1
    ck = set()
    for i, itemset in enumerate(lk[:-1]):
        for j in range(i+1, len(lk)):
            if len(set(itemset).intersection(set(lk[j]))) == leni:
                ck.add(tuple(sorted(set(itemset).union(set(lk[j])))))
    lk = [ itemset for itemset in ck if itemset_support(transactions, itemset) >= minsup ]
    while len(lk) > 0:
        yield lk
        leni = len(lk[0]) - 1
        ck = set()
        for i, itemset in enumerate(lk[:-1]):
            for j in range(i+1, len(lk)):
                if len(set(itemset).intersection(set(lk[j]))) == leni:
                    ck.add(tuple(sorted(set(itemset).union(set(lk[j])))))
        lk = [ itemset for itemset in ck if itemset_support(transactions, itemset) >= minsup ]

def unionize(lk: "list[set]"):
    for i, itemset in enumerate(lk[:-1]):
        leni = len(itemset)
        for j in range(i+1, len(lk)):
            if len(itemset.intersection(lk[j])) != leni - 1:
                continue
            print("union of a:", itemset, "b:", lk[j])
            yield itemset.union(lk[j])

def pairs(l: "list"):
    for i, itemset in enumerate(l[:-1]):
        for j in range(i+1, len(l)):
            yield itemset, l[j]

def rows_match(r1, r2)-> Literal[0, 1]:
    for item in r1:
        if item not in r2:
            return 0
    return 1

def itemset_support(transactions: "list[set]", itemset: set)-> float:
    count_s = sum(rows_match(itemset, transaction) for transaction in transactions)
    return count_s / len(transactions)

def itemset_confidence(transactions: "list[set]", lhs: set, rhs: set)-> float:
    itemset = lhs.union(rhs)
    count_s = sum(rows_match(itemset, transaction) for transaction in transactions)
    count_l = sum(rows_match(lhs, transaction) for transaction in transactions)
    return count_s / count_l

# I = TypeVar("I")
def most_common_item(l: list):
    item_labels = { i for i in l }
    items = { i: 0 for i in item_labels }
    for item in l:
        items[item] = items[item] + 1
    return dict_max_key(items)
