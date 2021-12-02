import numpy as np
import io


class Array:
    '''
    Save a 1-D list of compressed numpy-arrays in memory and decompress on demand
    '''
    def __init__(self):
        self.items = np.array([])
        self.size = 0
        self.shape = (0,)

    def append(self, numpy_array):
        self.items = np.append(self.items, Array.compress(numpy_array))
        self.size += 1
        self.shape = (self.size,)

    @staticmethod
    def compress(numpy_array):
        compressed_array = io.BytesIO()
        np.savez_compressed(compressed_array, numpy_array)
        return compressed_array

    @staticmethod
    def decompress(array):
        array.seek(0)
        decompressed_array = np.load(array)['arr_0']
        return decompressed_array

    @staticmethod
    def decompress_many(arrays):
        decompressed_arrays = []
        for n in arrays:
            decompressed_arrays.append(Array.decompress(n))
        return np.array(decompressed_arrays)

    def __getitem__(self, key):
        if type(key) is int:
            return Array.decompress(self.items[key])
        if type(key) is slice:
            return Array.decompress_many(self.items[key])
