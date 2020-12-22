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
        csv_reader = self.__read_csv()

        if not all(x in csv_reader.keys() for x in X) or not all(y in csv_reader.keys() for y in Y):
            raise Exception("Keys entered are not in the CSV file.")
        
        X_list = [csv_reader[x] for x in X]
        Y_list = [csv_reader[y] for y in Y]



        return np.array(X_list), np.array(Y_list)
    