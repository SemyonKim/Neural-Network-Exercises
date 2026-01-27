"""Модуль базовых алгоритмов линейной алгебры.
Задание состоит в том, чтобы имплементировать класс Matrix
(следует воспользоваться кодом из семинара ООП), учтя рекомендации pylint.
Для проверки кода следует использовать команду pylint matrix.py.
Pylint должен показывать 10 баллов.
Кроме того, следует добавить поддержку исключений в отмеченных местах.
Для проверки корректности алгоритмов следует сравнить результаты с соответствующими функциями numpy.
"""
# -*- coding: utf-8 -*-
import random
import copy
import numpy as np


class Matrix:
    """Custom Matrix class."""

    def __init__(self, nrows, ncols, init="zeros"):
        """Конструктор класса Matrix.
        Создаёт матрицу резмера nrows x ncols и инициализирует её методом init.
        nrows - количество строк матрицы
        ncols - количество столбцов матрицы
        init - метод инициализации элементов матрицы:
            "zeros" - инициализация нулями
            "ones" - инициализация единицами
            "random" - случайная инициализация
            "eye" - матрица с единицами на главной диагонали
        """
        if nrows < 0 or ncols < 0:
            raise ValueError("number of rows and cols must be positive")
        if init not in ["zeros", "ones", "random", "eye"]:
            raise ValueError(
                '"init" is different from "zeros", "ones", "eye" and "random"'
            )
        self.nrows = nrows
        self.ncols = ncols
        if init == "zeros":
            self.data = [[0 for row in range(self.ncols)] for col in range(self.nrows)]
        elif init == "ones":
            self.data = [[1 for row in range(self.ncols)] for col in range(self.nrows)]
        elif init == "random":
            self.data = [
                [random.random() for i in range(self.ncols)] for j in range(self.nrows)
            ]
        elif init == "eye" and self.ncols == self.nrows:
            self.data = [
                [1 if i == j else 0 for i in range(self.ncols)]
                for j in range(self.nrows)
            ]

    @staticmethod
    def from_dict(data):
        "Десеарилизация матрицы из словаря"
        ncols = data["ncols"]
        nrows = data["nrows"]
        items = data["data"]
        assert len(items) == ncols * nrows
        result = Matrix(nrows, ncols)
        for row in range(nrows):
            for col in range(ncols):
                result[(row, col)] = items[ncols * row + col]
        return result

    @staticmethod
    def to_dict(matr):
        "Сериализация матрицы в словарь"
        assert isinstance(matr, Matrix)
        nrows, ncols = matr.shape()
        data = []
        for row in range(nrows):
            for col in range(ncols):
                data.append(matr[(row, col)])
        return {"nrows": nrows, "ncols": ncols, "data": data}

    def __str__(self):
        return f"Matrix: {self.data}"

    def __repr__(self):
        return f"Matrix({self.nrows, self.ncols}, init=" ")"

    def shape(self):
        "Вернуть кортеж размера матрицы (nrows, ncols)"
        return (self.nrows, self.ncols)

    def __getitem__(self, index):
        """Получить элемент матрицы по индексу index
        index - список или кортеж, содержащий два элемента
        """

        if (type(index) not in [tuple, list]) and (len(index) != 2):
            raise ValueError(
                '"index" is not a tuple or list and does not contain two elements'
            )

        row, col = index
        if not (
            (-self.nrows <= row < self.nrows) and (-self.ncols <= col < self.ncols)
        ):
            raise ValueError('"index" outside of the matrix size')

        return self.data[row][col]

    def __setitem__(self, index, value):
        """Задать элемент матрицы по индексу index
        index - список или кортеж, содержащий два элемента
        value - Устанавливаемое значение
        """

        if (type(index) not in [tuple, list]) and (len(index) != 2):
            raise ValueError(
                '"index" is not a tuple or list and does not contain two elements'
            )

        row, col = index
        if not (
            (-self.nrows <= row < self.nrows) and (-self.ncols <= col < self.ncols)
        ):
            raise ValueError('"index" outside of the matrix size')

        self.data[row][col] = value

    def __sub__(self, rhs):
        "Вычесть матрицу rhs и вернуть результат"

        nrows, ncols = rhs.shape()
        if not (nrows == self.nrows and ncols == self.ncols):
            raise ValueError(
                '"rhs" size is different from the size of the current matrix'
            )

        result = [
            [a - b for a, b in zip(list_data, list_rhs)]
            for list_data, list_rhs in zip(self.data, rhs.data)
        ]
        tmp = Matrix(len(result), len(result[0]))
        tmp.data = result
        return tmp

    def __add__(self, rhs):
        "Сложить с матрицей rhs и вернуть результат"

        nrows, ncols = rhs.shape()
        if not (nrows == self.nrows and ncols == self.ncols):
            raise ValueError(
                '"rhs" size is different from the size of the current matrix'
            )

        result = [
            [a + b for a, b in zip(list_data, list_rhs)]
            for list_data, list_rhs in zip(self.data, rhs.data)
        ]
        tmp = Matrix(len(result), len(result[0]))
        tmp.data = result
        return tmp

    def __mul__(self, rhs):
        "Умножить на матрицу rhs и вернуть результат"

        nrows, ncols = rhs.shape()
        if not nrows == self.ncols:
            raise ValueError(
                "the number of rows rhs differs from the number of columns of the current matrix"
            )

        result = [[0 for row in range(ncols)] for col in range(self.nrows)]
        for i in range(self.nrows):
            for j in range(ncols):
                for k in range(nrows):
                    result[i][j] += self.data[i][k] * rhs.data[k][j]
        tmp = Matrix(len(result), len(result[0]))
        tmp.data = result
        return tmp

    def __pow__(self, power):
        "Возвести все элементы в степень pow и вернуть результат"

        result = [[i ** power for i in j] for j in self.data]
        tmp = Matrix(len(result), len(result[0]))
        tmp.data = result
        return tmp

    def sum(self):
        "Вернуть сумму всех элементов матрицы"

        return sum([sum(j) for j in self.data])

    def det(self):
        "Вычислить определитель матрицы"

        if not len(self.data) == len(self.data[0]):
            raise ArithmeticError("This matrix is not square")

        if len(self.data) == 3:
            detdt = (
                self.data[0][0] * self.data[1][1] * self.data[2][2]
                + self.data[2][0] * self.data[0][1] * self.data[1][2]
                + self.data[1][0] * self.data[2][1] * self.data[0][2]
            )
            detdt = detdt - (
                self.data[0][2] * self.data[1][1] * self.data[2][0]
                + self.data[0][0] * self.data[2][1] * self.data[1][2]
                + self.data[1][0] * self.data[0][1] * self.data[2][2]
            )
            return detdt
        if len(self.data) == 2:
            detdt = (
                self.data[0][0] * self.data[1][1] - self.data[1][0] * self.data[0][1]
            )
            return detdt
        if len(self.data) == 1:
            return self.data[0][0]
        tmp_a = copy.deepcopy(self.data)
        for diag, _ in enumerate(tmp_a):
            for i in range(diag + 1, len(tmp_a)):
                if tmp_a[diag][diag] == 0:
                    tmp_a[diag][diag] = 1.0e-15
                tmp = tmp_a[i][diag] / tmp_a[diag][diag]
                for j in range(len(tmp_a)):
                    tmp_a[i][j] = tmp_a[i][j] - tmp * tmp_a[diag][j]
                    # print(diag,i,j,tmp_a)
        detdt = 1.0
        for i, _ in enumerate(tmp_a):
            detdt = detdt * tmp_a[i][i]
        return detdt

    def transpose(self):
        "Транспонировать матрицу и вернуть результат"
        result = [
            [self.data[j][i] for j in range(len(self.data))]
            for i in range(len(self.data[0]))
        ]
        tmp = Matrix(len(result), len(result[0]))
        tmp.data = result
        return tmp

    def inv(self):
        "Вычислить обратную матрицу и вернуть результат"
        if not len(self.data) == len(self.data[0]):
            raise ArithmeticError("This matrix is not square")

        determinant = self.det()
        if determinant == 0:
            raise ArithmeticError("the determinant is zero")

        if len(self.data) == 2:
            return [
                [self.data[1][1] / determinant, -1 * self.data[0][1] / determinant],
                [-1 * self.data[1][0] / determinant, self.data[0][0] / determinant],
            ]

        tmp_a = copy.deepcopy(self.data)
        result = []
        for i, _ in enumerate(tmp_a):
            row_lst = []
            for j in range(len(tmp_a)):
                self.data = [
                    row[:j] + row[j + 1 :] for row in (tmp_a[:i] + tmp_a[i + 1 :])
                ]
                row_lst.append(((-1) ** (i + j)) * self.det())
            result.append(row_lst)
        self.data = copy.deepcopy(result)
        result = self.transpose()
        for i, _ in enumerate(result.data):
            for j in range(len(result.data)):
                result.data[i][j] = result.data[i][j] / determinant
        self.data = copy.deepcopy(tmp_a)
        return result

    def tonumpy(self):
        "Приведение к массиву numpy"
        return np.array(self.data)


def test():
    """test for Matrix class."""
    a_list = [
        [
            0.06714760250202245,
            0.7800177487213583,
            0.31517632731644396,
            0.46075367094201236,
        ],
        [
            0.9888266055943852,
            0.11901758603373436,
            0.009753276116418852,
            0.9382039930466721,
        ],
        [
            0.3611977756873682,
            0.11454068071948542,
            0.9936706951993024,
            0.25826790645406206,
        ],
    ]
    b_list = [
        [0.36014435363265973, 0.8387346177702184, 0.31200774387013863],
        [0.5671953226789971, 0.026895475083614895, 0.6328257656411127],
        [0.4851424323593291, 0.7789893952867298, 0.8637040316404491],
        [0.6987277464196714, 0.47548429906550627, 0.27171093938799806],
    ]
    a_array = np.array(a_list)
    b_array = np.array(b_list)
    a_dict = {
        "nrows": 3,
        "ncols": 4,
        "data": [item for sublist in a_list for item in sublist],
    }
    b_dict = {
        "nrows": 4,
        "ncols": 3,
        "data": [item for sublist in b_list for item in sublist],
    }

    a_matrix = Matrix.from_dict(a_dict)
    b_matrix = Matrix.from_dict(b_dict)

    # My_MatrixAlgo vs Numpy_MatrixAlgo
    # Pow's test:
    a_list = a_matrix ** 2
    b_list = b_matrix ** 3
    if not (
        np.allclose(a_list.tonumpy(), a_array ** 2)
        * np.allclose(b_list.tonumpy(), b_array ** 3)
    ):
        raise Exception("Pow's test failed!")

    # Sum's test:
    if not (a_matrix.sum() == np.sum(a_array)) * (b_matrix.sum() == np.sum(b_array)):
        raise Exception("Sum's test failed!")

    # Mul's test:
    a_list = a_matrix * b_matrix
    b_list = np.matmul(a_array, b_array)
    if not np.allclose(a_list.tonumpy(), b_list):
        raise Exception("Mul's test failed!")

    # Transpose's test:
    c_ttranspose = a_list.transpose()
    if not np.allclose(c_ttranspose.tonumpy(), b_list.transpose()):
        raise Exception("Transpose's test failed!")

    # Inversion's test:
    c_inv = a_list.inv()
    c_numpy_invert = np.linalg.inv(b_list)
    if not np.allclose(c_inv.tonumpy(), c_numpy_invert):
        raise Exception("Inversion's test failed!")

    # add/sub's test:
    e_ones = Matrix(c_inv.ncols, c_inv.nrows, init="eye")
    a_list = c_inv + e_ones
    b_list = c_inv - e_ones
    c_ttranspose = c_numpy_invert + e_ones.tonumpy()
    c_inv = c_numpy_invert - e_ones.tonumpy()
    if not (
        np.allclose(a_list.tonumpy(), c_ttranspose)
        * np.allclose(b_list.tonumpy(), c_inv)
    ):
        raise Exception("add/sub's test failed!")

    # Determinant's test:
    if not np.allclose(a_list.det(), np.linalg.det(c_ttranspose)):
        raise Exception("Determinant's test failed!")

    # Norm's test:
    if not (a_list ** 2).sum() ** 0.5 == np.linalg.norm(c_ttranspose):
        raise Exception("(Frobenius) Norm's test failed!")

    print("All tests passed successfully!")


if __name__ == "__main__":
    test()
