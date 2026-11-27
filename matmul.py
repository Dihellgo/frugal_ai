import matplotlib.pyplot as plt
import numpy as np
import time
from numba import jit
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.system("export OMP_NUM_THREADS=1")
os.system("export OPENBLAS_NUM_THREADS=1")
os.system("export MKL_NUM_THREADS=1")
os.system("export NUMEXPR_NUM_THREADS=1")

def matmul_naive(A, B, C, M, N, K):
    for i in range(M):
        for j in range(N):
            for k in range(K):
                C[i,j] += A[i,k]*B[k,j]
    return None

def matmul_naive_np(A, B, C, M, N, K):
    for i in range(M):
        for j in range(N):
            C[i,j] = np.sum(A[i,]*B[:,j])
    return None

@jit(nopython=True, cache=True, fastmath=False)
def matmul_naive_compiled(A, B, C, M, N, K):
    for i in range(M):
        for j in range(N):
            for k in range(K):
                C[i,j] += A[i,k]*B[k,j]
    return None

@jit(nopython=True, cache=True, fastmath=False)
def matmul_naive_np_compiled(A, B, C, M, N, K):
    for i in range(M):
        for j in range(N):
            C[i,j] = np.sum(A[i,]*B[:,j])
    return None

a=np.zeros((1,1), dtype="float32")
matmul_naive_compiled(a,a,a,1,1,1)
matmul_naive_np_compiled(a,a,a,1,1,1)

xpoints_naive = np.array(range(96,257,128))
xpoints = np.array(range(96,1921,256))
xpoints_fast = np.array(range(96,4097,128))
t_naive = []
t_naive_np = []
t_naive_compiled = []
t_naive_np_compiled = []
t_operator = []
t_matmul = []
t_dot = []

nb_iteration = 3

for m_size in xpoints_fast:
    print(m_size)
    M = m_size
    N = m_size
    K = m_size

    np.random.seed(0)

    A = ((np.random.random((M,K))-0.5)*0.1).astype("float32")
    B = ((np.random.random((K,N))-0.5)-0.1).astype("float32")
    C_verif = A@B

    for f,l in (
            (lambda A,B,*_ : A@B, t_operator),
            (lambda A,B,C,*_ : np.matmul(A,B,C), t_matmul),
            (lambda A,B,C,*_ : np.dot(A,B,C), t_dot)
        ):
        total_time = 0
        for _ in range(nb_iteration):
            C = np.zeros((M,N), dtype="float32")

            t_start = time.time()

            f(A, B, C, M, N, K)

            total_time += time.time() - t_start 
        l.append(total_time/nb_iteration) # In seconds

for m_size in xpoints:
    print(m_size)
    M = m_size
    N = m_size
    K = m_size

    np.random.seed(0)

    A = ((np.random.random((M,K))-0.5)*0.1).astype("float32")
    B = ((np.random.random((K,N))-0.5)-0.1).astype("float32")
    C_verif = A@B

    for f,l in (
            (matmul_naive_np,t_naive_np),
            (matmul_naive_compiled,t_naive_compiled),
            (matmul_naive_np_compiled,t_naive_np_compiled),
        ):
        total_time = 0
        for _ in range(nb_iteration):
            C = np.zeros((M,N), dtype="float32")

            t_start = time.time()

            f(A, B, C, M, N, K)

            total_time += time.time() - t_start 
        l.append(total_time/nb_iteration) # In seconds

for m_size in xpoints_naive:
    print(m_size)
    M = m_size
    N = m_size
    K = m_size

    np.random.seed(0)

    A = ((np.random.random((M,K))-0.5)*0.1).astype("float32")
    B = ((np.random.random((K,N))-0.5)-0.1).astype("float32")
    C_verif = A@B

    total_time = 0
    for _ in range(nb_iteration):
        C = np.zeros((M,N), dtype="float32")

        t_start = time.time()

        matmul_naive(A, B, C, M, N, K)

        total_time += time.time() - t_start 
    t_naive.append(total_time) # In seconds

y_naive = xpoints_naive**3/t_naive
y_naive_np = xpoints**3/t_naive_np
y_naive_compiled = xpoints**3/t_naive_compiled
y_naive_np_compiled = xpoints**3/t_naive_np_compiled
y_operator = xpoints_fast**3/t_operator
y_matmul = xpoints_fast**3/t_matmul
y_dot = xpoints_fast**3/t_dot

fig = plt.figure(figsize=(12,8))
plt.plot(xpoints_naive,y_naive, label = "naive")
plt.plot(xpoints,y_naive_np, label="vectorized")
plt.plot(xpoints,y_naive_compiled, label="compiled")
plt.plot(xpoints,y_naive_np_compiled, label="vectorized compiled")
plt.plot(xpoints_fast,y_operator, label="@")
plt.plot(xpoints_fast,y_matmul, label="matmul")
plt.plot(xpoints_fast,y_dot, label="dot")

plt.xlabel("Matrix size")
plt.ylabel("FLOPS")
plt.legend()
plt.savefig("FLOPSall.png")
plt.show()
# print(ellapsed_time)
# 
# if(np.sum((C - C_verif)**2) > 0.00001):
# 	print("Error: results do not match!")
# 	exit()
