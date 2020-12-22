import unittest
from chapter_01 import read_datasets

#Test read_datasets.py

dataset_reader_inst = read_datasets.DatasetReader("datasets/boston_housing.csv", ",")


class TestDatasetReader(unittest.TestCase):

    def test_dsr_get_col(self):
        col_list = dataset_reader_inst.get_columns()
        self.assertEqual(col_list[0], "crim")

    def test_x_y_shape(self):
        X, y = dataset_reader_inst.read_into_array(["crim", "indus"], ["medv"])

        self.assertEqual(X.shape, (2, 506))
        self.assertEqual(y.shape, (1, 506))

    def test_x_y_assert(self):
        with self.assertRaises(Exception):
            X, y = dataset_reader_inst.read_into_array(["hello", "indus"], ["dev"])

if __name__ == '__main__':
    unittest.main()

