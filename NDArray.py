import array
import math
from collections.abc import Iterable
import copy

class NDArray():

    def __init__(self, shape, type_='i', fill=0):
        if not isinstance(shape, Iterable):
            shape = [shape]
        self.shape = shape
        self.size = math.prod(shape)
        if isinstance(fill, (int, float, complex)):
            fill = [fill] * self.size
        else:
            assert len(fill) == self.size
        self._array = array.array(type_, fill)

    @classmethod
    def ones(cls, shape, type_='i'):
        return cls(shape, type_, fill=1)

    @property
    def ndim(self):
        return len(self.shape)

    def __getitem__(self, indexes):
        if  self.ndim == 1:
            if isinstance(indexes, slice):
                values = self._array[indexes]
                return NDArray(len(values), fill=values)
            else:
                index = self._flatten_indexes(indexes)  # 
                return self._array[index]
        else:
            if isinstance(indexes, int):
                start = indexes * self.shape[1] #
                stop = start + self.shape[1]
                values = self._array[start:stop]
                return NDArray(len(values), fill=values)
            elif isinstance(indexes, slice):
                rows = []
                count = 0
                start = indexes.start
                if start is None:
                    start = 0
                stop = indexes.stop
                if stop is None:
                    stop = 0
                step = indexes.step
                if step is None:
                    step = 1
                for i in range(start, stop, step):
                    rows.extend([value for value in self[i]])
                    count += 1
                return NDArray((count, self.shape[1]), fill=rows)
            elif len(indexes) == 2:
                if isinstance(indexes[0], slice) and isinstance(indexes[1], int):
                    rows = self[indexes[0]]
                    values = []
                    for row in rows:
                        values.append(row[indexes[1]])
                    return NDArray(len(values), fill=values)
                elif isinstance(indexes[0], int) and isinstance(indexes[1], slice):
                    pass    #

    def __iter__(self):
        for i in range(self.shape[0]):
            yield self[i]

    def T(self):
        if (self.shape[0] == 1):
            return NDArray(self.shape, fill=self._array)
        else:
            rows = []
            for count_columns in range(self.shape[1]):
                rows.extend(self[0:self.shape[0]:1, count_columns])
            return NDArray((self.shape[1], self.shape[0]), fill=rows)
        
    def __setitem__(self, indexes, value):
        index = self._flatten_indexes(indexes)
        self._array[index] = value

    def _flatten_indexes(self, indexes):
        if self.ndim == 1:
            return indexes
        if self.ndim == 2:
            return indexes[0] * self.shape[1] + indexes[1]

    def __element_wise_operator(self, other, operator):
        if self.shape != other.shape:
            raise RuntimeError(f"{self.shape} != {other.shape}")
        result = NDArray(self.shape, self._array.typecode)

        with self as a1, other as a2, result as a3:
            for i in range(a1.size):
                a3[i] = operator(a1[i], a2[i])
        return result

    def __add__(self, other):
        return self.__element_wise_operator(other, lambda a,b: a+b)
    
    def __sub__(self, other):
        return self.__element_wise_operator(other, lambda a,b: a-b)

    def __mul__(self, other):
        if (self.ndim != 1):
            result = (self.__matmul__(other))
        else:
            assert(self.shape == other.shape)
            result = 0
            for i in range(self.shape[0]):
                result += self[i] * other[i]
        return result

    def __matmul__(self, other):
       if (self.ndim != 2 and other.ndim != 2):
           raise RuntimeError("Not supported")
       assert(self.shape[1] == other.shape[0])
       result = []
       for counter_rows in range(self.shape[0]):
           for counter_cols in range(other.shape[1]):
               result += [self[counter_rows] * other[0:other.shape[0]:1, counter_cols]]
       return NDArray((self.shape[0], other.shape[1]), fill=result)

    def __enter__(self):
        self._shape = self.shape
        self.shape = [self.size]
        return self

    def __exit__(self, *args):
        self.shape = self._shape

    def flatten(self):
        return self._array

    def __str__(self):
        shape = None
        if self.ndim == 1:
            shape = [1, ] + self.shape
        elif self.ndim == 2:
            shape = self.shape
        else:
            raise RuntimeError("Not supported")
        s = []
        for i in range(shape[0]):
            row = self._array[i * shape[1]: 
                                i * shape[1]+shape[1]]
            s.append(" ".join(map(str, row)))
        return "\n".join(s)

if __name__ == "__main__":
    arr = NDArray((3,4), fill=list(range(12)))
    print(arr)
    arr = arr.T()
    print("-----Transposed-----")
    print(arr)
    print("-----Multiplication-----")
    res = arr[0] * arr[1]
    print(res)
    print("-----Matrix multiplication-----")
    arr_1 = NDArray((3,3), fill=list(range(9)))
    arr_2 = NDArray((3,2), fill=list(range(6)))
    print("Matrix 1")
    print(arr_1)
    print("Matrix 2")
    print(arr_2)
    print("--------res--------")
    res = arr_1 * arr_2
    print(res)