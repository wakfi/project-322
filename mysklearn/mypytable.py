import copy
import csv
from tabulate import tabulate

class MyPyTable:
    """Represents a 2D table of data with column names.

    Attributes:
        column_names(list of str): M column names
        data(list of list of obj): 2D data structure storing mixed type data. There are N rows by M columns.
    """

    def __init__(self, column_names=None, data=None):
        """Initializer for MyPyTable.

        Args:
            column_names(list of str): initial M column names (None if empty)
            data(list of list of obj): initial table data in shape NxM (None if empty)
        """
        if column_names is None:
            column_names = []
        self.column_names = copy.deepcopy(column_names)
        if data is None:
            data = []
        self.data = copy.deepcopy(data)

    def pretty_print(self, rows=None):
        """Prints the table in a nicely formatted grid structure.
        """
        if rows is None:
            rows = len(self.data)
        print(tabulate(self.data[:rows], headers=self.column_names))

    def get_shape(self):
        """Computes the dimension of the table (N x M).

        Returns:
            int: number of rows in the table (N)
            int: number of cols in the table (M)
        """
        return len(self.data), len(self.column_names)

    def get_column(self, col_identifier, include_missing_values=True):
        """Extracts a column from the table data as a list.

        Args:
            col_identifier(str or int): string for a column name or int
                for a column index
            include_missing_values(bool): True if missing values ("NA")
                should be included in the column, False otherwise.

        Returns:
            list of obj: 1D list of values in the column
        Notes:
            Raise ValueError on invalid col_identifier
        """
        if isinstance(col_identifier, str):
            col = self.column_names.index(col_identifier)
        else:
            col = col_identifier
        if include_missing_values:
            return [ row[col] for row in self.data ]
        return [ row[col] for row in self.data if row[col] != 'NA']

    def apply_to_column(self, col_identifier, operation):
        if isinstance(col_identifier, str):
            col = self.column_names.index(col_identifier)
        else:
            col = col_identifier
        c = self.get_column(col)
        for i, result in enumerate(operation(c)):
            self.data[i][col] = result

    def convert_to_numeric(self):
        """Try to convert each value in the table to a numeric type (float).

        Notes:
            Leave values as is that cannot be converted to numeric.
        """
        for i in range(len(self.data)):
            for j in range(len(self.column_names)):
                try:
                    self.data[i][j] = float(self.data[i][j])
                except ValueError:
                    pass


    def drop_rows(self, rows_to_drop):
        """Remove rows from the table data.

        Args:
            rows_to_drop(list of list of obj): list of rows to remove from the table data.
        """
        self.data = [ row for row in self.data if rows_to_drop.count(row) == 0 ]

    def load_from_file(self, filename):
        """Load column names and data from a CSV file.

        Args:
            filename(str): relative path for the CSV file to open and load the contents of.

        Returns:
            MyPyTable: return self so the caller can write code like: table = MyPyTable().load_from_file(fname)

        Notes:
            Use the csv module.
            First row of CSV file is assumed to be the header.
            Calls convert_to_numeric() after load
        """
        with open(filename, newline='') as csvfile:
            reader = csv.reader(csvfile)
            self.data = [ row for row in reader ]
            self.column_names = self.data.pop(0)
        return self

    def save_to_file(self, filename):
        """Save column names and data to a CSV file.

        Args:
            filename(str): relative path for the CSV file to save the contents to.

        Notes:
            Use the csv module.
        """
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(self.column_names)
            for row in self.data:
                writer.writerow(row)

    def find_duplicates(self, key_column_names):
        """Returns a list of duplicates. Rows are identified uniquely based on key_column_names.

        Args:
            key_column_names(list of str): column names to use as row keys.

        Returns:
            list of list of obj: list of duplicate rows found

        Notes:
            Subsequent occurrence(s) of a row are considered the duplicate(s). The first instance is not
                considered a duplicate.
        """
        uniqueRecords = []
        duplicates = []
        key_cols = [ self.column_names.index(key_column_name) for key_column_name in key_column_names ] # raises ValueError if any key_column_name is not found
        for row in self.data:
            key = [ row[key_col] for key_col in key_cols ]
            if uniqueRecords.count(key) == 0:
                uniqueRecords.append(key)
            else:
                duplicates.append(row)
        return duplicates

    def remove_rows_with_missing_values(self):
        """Remove rows from the table data that contain a missing value ("NA").
        """
        self.data = [ row for row in self.data if row.count('NA') == 0 ]

    def replace_missing_values_with_column_average(self, col_name):
        """For columns with continuous data, fill missing values in a column by the column's original average.

        Args:
            col_name(str): name of column to fill with the original average (of the column).
        """
        col = self.column_names.index(col_name)
        column = self.get_column(col_name)
        avg = 0
        count = 0
        for item in column:
            if item != 'NA':
                avg += item
                count += 1
        if count == len(column):
            return
        avg = avg / count
        for row in range(len(self.data)):
            if self.data[row][col] == 'NA':
                self.data[row][col] = avg


    def compute_summary_statistics(self, col_names):
        """Calculates summary stats for this MyPyTable and stores the stats in a new MyPyTable.

        Args:
            col_names(list of str): names of the continuous columns to compute summary stats for.

        Returns:
            MyPyTable: stores the summary stats computed. The column names and their order is as follows: ["attribute", "min", "max", "mid", "avg", "median"]
        """
        table = []
        table_col_names = ['attribute', 'min', 'max', 'mid', 'avg', 'median']
        for col_name in col_names:
            col = self.get_column(col_name)
            col = [ val for val in col if val != 'NA' ]
            if len(col) == 0:
                continue
            col.sort()
            row = [col_name, min(col), max(col)] # attribute, min, max
            row.extend([(row[1] + row[2]) / 2, sum(col) / len(col), col[round(len(col)/2) - 1] if len(col) % 2 == 1  else (col[round(len(col)/2) - 1] + col[round(len(col)/2)]) / 2]) # mid, avg, median
            table.append(row)
        return MyPyTable(table_col_names, table)

    def perform_inner_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable inner joined with other_table based on key_column_names.

        Args:
            other_table(MyPyTable): the second table to join this table with.
            key_column_names(list of str): column names to use as row keys.

        Returns:
            MyPyTable: the inner joined table.
        """
        attr_names = [ *self.column_names ]
        attr_names.extend([ name for name in other_table.column_names if attr_names.count(name) == 0 ])
        table = []
        self_key_cols = [ self.column_names.index(key_column_name) for key_column_name in key_column_names ] # raises ValueError if any key_column_name is not found
        self_nonkey_cols = [ j for j in range(len(self.column_names)) if self_key_cols.count(j) == 0 ]
        other_key_cols = [ other_table.column_names.index(key_column_name) for key_column_name in key_column_names ] # raises ValueError if any key_column_name is not found
        other_nonkey_cols = [ j for j in range(len(other_table.column_names)) if other_key_cols.count(j) == 0 ]
        other_keys = []
        for row in other_table.data:
            other_key = [ row[key_col] for key_col in other_key_cols ]
            if other_keys.count(other_key) == 0:
                other_keys.append(other_key)
            else:
                other_keys.append('NA') # placeholding in case of duplicate keys for some reason
        for row in self.data:
            self_key = [ row[key_col] for key_col in self_key_cols ]
            if other_keys.count(self_key) != 0:
                other_row = other_table.data[other_keys.index(self_key)]
                table.append([*[ row[col] for col in range(len(self.column_names)) ], *[ other_row[col] for col in other_nonkey_cols ]])
        return MyPyTable(copy.deepcopy(attr_names), copy.deepcopy(table))

    def perform_full_outer_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable fully outer joined with other_table based on key_column_names.

        Args:
            other_table(MyPyTable): the second table to join this table with.
            key_column_names(list of str): column names to use as row keys.

        Returns:
            MyPyTable: the fully outer joined table.

        Notes:
            Pad the attributes with missing values with "NA".
        """
        attr_names = [ *self.column_names ]
        attr_names.extend([ name for name in other_table.column_names if attr_names.count(name) == 0 ])
        table = []
        self_key_cols = [ self.column_names.index(key_column_name) for key_column_name in key_column_names ] # raises ValueError if any key_column_name is not found
        self_nonkey_cols = [ j for j in range(len(self.column_names)) if self_key_cols.count(j) == 0 ]
        other_key_cols = [ other_table.column_names.index(key_column_name) for key_column_name in key_column_names ] # raises ValueError if any key_column_name is not found
        other_nonkey_cols = [ j for j in range(len(other_table.column_names)) if other_key_cols.count(j) == 0 ]
        self_keys = []
        other_keys = []
        self_na = [ True if i in self_key_cols or i >= len(self.column_names) else False for i in range(len(attr_names)) ]
        for key_column_name in key_column_names:
            self_na[self.column_names.index(key_column_name)] = other_table.column_names.index(key_column_name)
        self_na = [ other_table.column_names.index(attr_names[i]) if self_na[i] is True else self_na[i] for i in range(len(self_na)) ]
        self_na = [ 'NA' if val is False else val for val in self_na ]
        other_na = [ 'NA' for i in range(len(other_nonkey_cols)) ]
        for row in other_table.data:
            other_key = [ row[key_col] for key_col in other_key_cols ]
            if other_keys.count(other_key) == 0:
                other_keys.append(other_key)
            else:
                other_keys.append('NA') # placeholding in case of duplicate keys for some reason
        for row in self.data:
            self_key = [ row[key_col] for key_col in self_key_cols ]
            self_keys.append(self_key)
            if other_keys.count(self_key) != 0:
                other_row = other_table.data[other_keys.index(self_key)]
                table.append([*[ row[col] for col in range(len(self.column_names)) ], *[ other_row[col] for col in other_nonkey_cols ]])
            else:
                table.append([*[ row[col] for col in range(len(self.column_names)) ], *other_na])
        for row in other_table.data:
            other_key = [ row[key_col] for key_col in other_key_cols ]
            if self_keys.count(other_key) == 0:
                table.append([ row[col] if col != 'NA' else col for col in self_na ])
        return MyPyTable(copy.deepcopy(attr_names), copy.deepcopy(table))
