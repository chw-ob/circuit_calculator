from sympy import symbols, Eq, Matrix, solve
import numpy as np
import math
class component():
    def __init__(self, net=[], property=None):
        self.net = net
        self.property = property
    def get_kind(self):
        return None
    def get_n(self, List):
        for i in self.net:
            List.add(i)
        return List

    def get_matrix_A(self, A, i):
        A[self.net[0], i] = 1
        A[self.net[1], i] = -1
        return A

    def get_vector_Usb(self,Usb,i):
        return Usb

    def get_vector_Ub(self,Ub,i):
        return Ub
        pass

    def get_vector_Zb(self,Zb,i):
        return Zb
        pass

    def get_vector_Ib(self,Ib,i):
        return Ib
        pass

    def get_vector_Isb(self,Isb,i):
        return Isb
        pass

    def get_matrix_F(self,F,i):
        return F
        pass

    def get_matrix_H(self,H,i):
        return H
        pass

    def get_vector_Un(self,Un,i):
        return Un
        pass
class component_R(component):
    def get_vector_Zb(self, Zb, i):
        Zb[i, i] = self.property
        return Zb

    def get_matrix_F(self, F, i):
        F[i, i] = -1
        return F

    def get_matrix_H(self, H, i):
        H[i, i] = self.property
        return H
class component_source_v(component):
    def get_kind(self):
        return "source_v"
    def get_vector_Usb(self, Usb, i):
        Usb[i, 0] = -self.property
        return Usb

    def get_matrix_F(self, F, i):
        F[i, i] = 1
class component_Y(component):
    def get_matrix_H(self,H,i):
        H[i,i]=-1
    def get_matrix_F(self,F,i):
        F[i,i]=self.property
class component_source_i(component):
    def get_matrix_H(self,H,i):
        H[i,i]=1
    def get_vector_Isb(self,Isb,i):
        Isb[i,0]=self.property
class component_C(component):
    pass
class component_L(component):
    pass
class calculator():
    def __init__(self):
        pass

    @staticmethod
    def get_n(e):
        List = set()
        for i in e:
            i.get_n(List)
        return len(List)

    @staticmethod
    def get_b(e):
        return len(e)

    @staticmethod
    def get_matrix_A(e):
        n, b = calculator.get_n(e), calculator.get_b(e)
        A = np.zeros([n, b])
        for i in range(len(e)):
           e[i].get_matrix_A(A,i)
        return A

    @staticmethod
    def get_vector_Usb(e):
        b = calculator.get_b(e)
        Usb = np.zeros([b, 1])
        for i in range(len(e)):
            e[i].get_vector_Usb(Usb,i)
        return Matrix(Usb)

    @staticmethod
    def get_vector_Ub(e):
        b = calculator.get_b(e)
        Str = ''
        for i in range(b):
            Str += "U" + str(i) + ' '
        ub = Matrix(symbols(Str))
        return ub

    @staticmethod
    def get_vector_Zb(e):
        b = calculator.get_b(e)
        Zb = np.zeros([b, b])
        for i in range(len(e)):
            e[i].get_matrix_Zb(Zb,i)
        return Matrix(Zb)

    @staticmethod
    def get_vector_Ib(e):
        b = calculator.get_b(e)
        Str = ''
        for i in range(b):
            Str += "I" + str(i) + ' '
        Ib = Matrix(symbols(Str))
        return Ib

    @staticmethod
    def get_vector_Isb(e):
        b = calculator.get_b(e)
        Isb=np.zeros([b, 1])
        for i in range(len(e)):
            e[i].get_vector_Isb(Isb,i)
        return Matrix(Isb)

    @staticmethod
    def get_matrix_F(e):
        b = calculator.get_b(e)
        F = np.zeros([b, b])
        for i in range(len(e)):
            e[i].get_matrix_F(F,i)
        return F

    @staticmethod
    def get_matrix_H(e):
        b = calculator.get_b(e)
        H = np.zeros([b, b])
        for i in range(len(e)):
            e[i].get_matrix_H(H,i)
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
        while i < len(e):
            if e[i].get_kind() == "source_v":
                e.pop(i)
                continue
            i += 1
        e.append(component_source_v([net_1,net_2],1))
        solution = calculator.get_linear_solution(e)
        Ib = calculator.get_vector_Ib(e)
        i = solution[Ib[len(e) - 1]]
        return 1 / i
        pass

    @staticmethod
    def X_t(A,B,x,t):
        return A.multiply(x)+B.multiply(t)

    @staticmethod
    def cal(step, A, B, us, init, t):
        L1 = calculator.X_t(A, B, init, us)
        L2 = calculator.X_t(A, B, init + L1 * step / 2, Matrix([math.sin(t + step / 2)]))
        L3 = calculator.X_t(A, B, init + L2 * step / 2, Matrix([math.sin(t + step / 2)]))
        L4 = calculator.X_t(A, B, init + L3 * step, Matrix([math.sin(t + step)]))
        delta = (L1 + L2 * 2 + L3 * 2 + L4) / 6 * step
        return init + delta

    @staticmethod
    def get_state_A(e_temp):

        pass

    @staticmethod
    def get_state_B(e_temp):
        pass

    @staticmethod
    def get_init_value(e_temp):
        pass

    @staticmethod
    def get_sti_value(e_temp):
        pass


if __name__ == "__main__":
    e1=component_source_v([0,1],15)
    e2=component_R([1,2],10)
    e3=component_R([1,2],10)
    e4 = component_R([2, 0], 20)
    e5 = component_R([2, 0], 20)
    e6=component_Y([2,0],0.05)
    e7 = component_Y([1, 2], 0.1)
    # e6= ["source_v", [0, 0], [0, 2], 20]
    e = [e1, e2, e3, e4,e5, e6,e7]

    ca = calculator()
    print(calculator.get_linear_solution(e))
