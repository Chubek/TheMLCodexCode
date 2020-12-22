import csv
import numpy as np

class DatasetReader:

    def __init__(self, path, delimiter):
        self.path = path
        self.delimiter = delimiter


    def __read_csv(self):
        csv_file =  open(self.path, 'r')
        csv_reader = csv.DictReader(csv_file, delimiter=self.delimiter)
        ret_dict = {}
        for row in csv_reader:
            for fieldname in csv_reader.fieldnames:
                if fieldname in ret_dict:
                    ret_dict[fieldname].append(row[fieldname])
                else:
                    ret_dict[fieldname] = [row[fieldname]] 
        csv_file.close()
        return ret_dict

    def get_columns(self):
        csv_reader = self.__read_csv()
        return list(csv_reader.keys())


    def read_into_array(self, X, Y):
        if not X in csv_reader.row.keys() or not Y in csv_reader.row.keys():
            raise Exception("Keys entered are not in the CSV file.")
        if not len(Y) == 1:
            raise Exception("Length of Y should be 1 --- Target is invariable one column.")

        csv_reader = self.__read_csv()

        X_list = []

        for x in X:
            for row in csv_reader:
                X_list.append(row[x])

        Y_list = []

        for y in Y:
            for row in csv_reader:
                Y_list.append(row[y])


        return np.array(X_list), np.array(Y_list)
    