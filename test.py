e1 = ["source_v", [0, 0], [0, 1], 6]
e2 = ["r", [0, 0], [1, 2], 6]
#e3 = ["r", [0, 0], [2, 0], 6]
#e4 = ["r", [0, 0], [2, 0], 6]
e=[e1,e2]
import numpy as np
from sympy import symbols,Eq,Matrix
def get_n(e):
    List = set()
    for i in e:
        for j in i[2]:
            List.add(j)
    return len(List)
def get_b(e):
    return len(e)
def get_matrix_A(e):
    n,b=get_n(e),get_b(e)
    A=np.zeros([n,b])
    for i in range(len(e)):
        A[e[i][2][0],i]=1
        A[e[i][2][1], i] = -1
    return A
def get_vector_Usb(e):
    b =get_b(e)
    Usb=np.zeros([b,1])
    for i in range(len(e)):
        if e[i][0]=="source_v":
            Usb[i,0]=-e[i][3]
    return Matrix(Usb)
def get_vector_Ub(e):
    b = get_b(e)
    Str = ''
    for i in range(b):
        Str += "U" + str(i) + ' '
    ub = Matrix(symbols(Str))
    return ub
def get_vector_Zb(e):
    b=get_b(e)
    Zb=np.zeros([b,b])
    for i in range(b):
        if e[i][0]=="r":
            Zb[i,i]=e[i][3]
    return Matrix(Zb)
def get_vector_Ib(e):
    b=get_b(e)
    Str=''
    for i in range(b):
        Str+="I"+str(i)+' '
    print(symbols(Str))
    Ib=Matrix(symbols(Str))
    print(Ib)
    return Ib
def get_vector_Isb(e):
    b=get_b(e)
    return Matrix(np.zeros([b,1]))
def get_matrix_F(e):
    b=get_b(e)
    F=np.zeros([b,b])
    for i in range(len(e)):
        if e[i][0]=="source_v":
            F[i,i]=1
        if e[i][0]=="r":
            F[i,i]=-1
    return F
def get_matrix_H(e):
    b=get_b(e)
    H=np.zeros([b,b])
    for i in range(len(e)):
        if e[i][0]=="source_v":
            H[i,i]=0
        if e[i][0]=="r":
            H[i,i]=e[i][3]
            pass
    return H
Ub=get_vector_Ub(e)
F=Matrix(get_matrix_F(e))
H=Matrix(get_matrix_H(e))
Ib=get_vector_Ib(e)
Ub=get_vector_Ub(e)
Usb=get_vector_Usb(e)
Isb=get_vector_Isb(e)
A=Matrix(get_matrix_A(e))
def get_vector_Un(e):
    n=get_n(e)
    Str=""
    for i in range(n):
        Str += "Un" + str(i) + ' '
    Un=Matrix(symbols(Str))
    Un[0]=0
    return Un
Un=get_vector_Un(e)
from sympy import solve
eq1 = Eq(A * Ib, Matrix(np.zeros([A.shape[0], 1])))
print(A.multiply(Ib))
At = A.transpose()
eq2 = Eq(At * Un, Ub)
eq3 = Eq(F * Ub + H * Ib, Usb + Isb)
solution = solve([eq1, eq2, eq3], [Ib, Un, Ub])


#print(Un,Ub)
