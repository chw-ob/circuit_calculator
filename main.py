import numpy as np
from sympy import symbols, Eq, Matrix, solve


class calculator():
    def __init__(self):
        pass
    @staticmethod
    def get_n(e):
        List = set()
        for i in e:
            for j in i[2]:
                List.add(j)
        return len(List)

    @staticmethod
    def get_b(e):
        return len(e)

    @staticmethod
    def get_matrix_A( e):
        n, b = calculator.get_n(e), calculator.get_b(e)
        A = np.zeros([n, b])
        for i in range(len(e)):
            A[e[i][2][0], i] = 1
            A[e[i][2][1], i] = -1
        return A

    @staticmethod
    def get_vector_Usb(e):
        b = calculator.get_b(e)
        Usb = np.zeros([b, 1])
        for i in range(len(e)):
            if e[i][0] == "source_v":
                Usb[i, 0] = -e[i][3]
        return Matrix(Usb)

    @staticmethod
    def get_vector_Ub( e):
        b = calculator.get_b(e)
        Str = ''
        for i in range(b):
            Str += "U" + str(i) + ' '
        ub = Matrix(symbols(Str))
        return ub

    @staticmethod
    def get_vector_Zb( e):
        b =calculator.get_b(e)
        Zb = np.zeros([b, b])
        for i in range(b):
            if e[i][0] == "r":
                Zb[i, i] = e[i][3]
        return Matrix(Zb)

    @staticmethod
    def get_vector_Ib(e):
        b =calculator.get_b(e)
        Str = ''
        for i in range(b):
            Str += "I" + str(i) + ' '
        Ib = Matrix(symbols(Str))
        return Ib

    @staticmethod
    def get_vector_Isb( e):
        b =calculator.get_b(e)
        return Matrix(np.zeros([b, 1]))

    @staticmethod
    def get_matrix_F( e):
        b = calculator.get_b(e)
        F = np.zeros([b, b])
        for i in range(len(e)):
            if e[i][0] == "source_v":
                F[i, i] = 1
            if e[i][0] == "r":
                F[i, i] = -1
        return F

    @staticmethod
    def get_matrix_H(e):
        b = calculator.get_b(e)
        H = np.zeros([b, b])
        for i in range(len(e)):
            if e[i][0] == "source_v":
                H[i, i] = 0
            if e[i][0] == "r":
                H[i, i] = e[i][3]
                pass
        return H

    @staticmethod
    def get_vector_Un(e):
        n = calculator.get_n(e)
        Str = ""
        for i in range(n):
            Str += "Un" + str(i) + ' '
        Un = Matrix(symbols(Str))
        Un[0] = 0
        return Un

    @staticmethod
    def get_linear_solution(e):
        F = Matrix(calculator.get_matrix_F(e))
        H = Matrix(calculator.get_matrix_H(e))
        Ib = calculator.get_vector_Ib(e)
        Ub = calculator.get_vector_Ub(e)
        Usb = calculator.get_vector_Usb(e)
        Isb = calculator.get_vector_Isb(e)
        Un = calculator.get_vector_Un(e)
        A = Matrix(calculator.get_matrix_A(e))
        At = A.transpose()
        eq1, eq2, eq3 = A.multiply(Ib), At.multiply(Un), F.multiply(Ub) + H.multiply(Ib)
        eq = Matrix([[eq1], [eq2], [eq3]])
        eq1_, eq2_, eq3_ = Matrix(np.zeros([eq1.shape[0], 1])), Ub, Usb + Isb
        eq_ = Matrix([[eq1_], [eq2_], [eq3_]])
        eq = Eq(eq, eq_)
        X = Matrix([Ib, Un, Ub])
        solution = solve(eq, X)
        return solution

    @staticmethod
    def get_equal_R(e_temp, net_1, net_2):
        import copy
        e = copy.deepcopy(e_temp)
        i = 0
        '''while i <len(e):
            if e[i][0]=="source_v":
                e[i][3]=0
            i+=1'''
        while i < len(e):
            if e[i][0] == "source_v":
                e.pop(i)
                continue
            i += 1
        e.append(["source_v", [0, 0], [net_1, net_2], 1])
        solution = calculator.get_linear_solution(e)
        Ib = calculator.get_vector_Ib(e)
        i = solution[Ib[len(e) - 1]]
        return 1 / i
        pass


if __name__ == "__main__":
    e1 = ["source_v", [0, 0], [0, 1], 15]
    e2 = ["r", [0, 0], [1, 2], 10]
    e3 = ["r", [0, 0], [1, 2], 10]
    e4 = ["r", [0, 0], [0, 2], 20]
    e5 = ["r", [0, 0], [0, 2], 20]
    # e6= ["source_v", [0, 0], [0, 2], 20]
    e = [e1, e2, e3, e4, e5]
    ca = calculator()
    print(calculator.get_linear_solution(e))
    print(ca.get_equal_R(e, 0, 1))
