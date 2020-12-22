import unittest
from chapter_01 import read_datasets

#Test read_datasets.py

dataset_reader_inst = read_datasets.DatasetReader("datasets/boston_housing.csv", ",")


class TestDatasetReader(unittest.TestCase):

    def test_dsr_get_col(self):
        col_list = dataset_reader_inst.get_columns()

        self.assertEqual(col_list[0], "crim")




if __name__ == '__main__':
    unittest.main()

