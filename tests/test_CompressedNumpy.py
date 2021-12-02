import numpy as np
import VCWA.CompressedNumpy as cnp


def test_simple_insert():
    compressed_np = cnp.Array()
    numpy_array = np.array([[1, 2], [3, 4]])
    compressed_np.append(numpy_array)
    assert compressed_np.size == 1


def test_decompress():
    compressed_np = cnp.Array()
    numpy_array = np.array([[1, 2], [3, 4]])
    compressed_np.append(numpy_array)
    assert (cnp.Array.decompress(compressed_np.items[0]) == numpy_array).all()


def test_getitem():
    compressed_np = cnp.Array()
    numpy_array = np.array([[1, 2], [3, 4]])
    compressed_np.append(numpy_array)
    assert (compressed_np[0] == numpy_array).all()


def test_getitem_range():
    compressed_np = cnp.Array()
    for i in range(10):
        numpy_array = np.array([1, 2, 3, 4])
        compressed_np.append(numpy_array)
    assert (compressed_np[2:5] == np.array([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]])).all()
