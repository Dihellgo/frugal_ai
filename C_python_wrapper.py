import os, sys
import matplotlib.pyplot as plt
import numpy as np

size = np.array(range(960,9600,960))

os.environ["OMP_NUM_THREADS"] = "8"
os.environ["OPENBLAS_NUM_THREADS"] = "8"



l_size = 9600

if os.system("gcc matmul_vect.c -o matmul -O2 -fopenmp -ffast-math -funroll-loops -march=native -lm -lopenblas"): sys.exit("Compilation error")


t = 0
y = 0
for _ in range(10):
    c_code_output = os.popen("./matmul %d"%(l_size)).read()

    if "ERROR ! MATRIX DIFF !" in c_code_output:
        print(c_code_output)
        exit()

    openblas_time, custom_time = [float(line.split()[-2]) for line in c_code_output.splitlines()]

    print ("OpenBLAS at size %d \t time %f s"%(l_size, openblas_time))
    print ("Custom at size %d \t time %f s"%(l_size, custom_time))
    t+= custom_time
    y+= openblas_time
print(l_size**3*(10/y)*10**(-9), "GFLOPS")
ref_noopen=t/3 
print(f"custom : {t/3:.2f}s")
print(f"openBlas : {y/3:.2f}s")
print(f"{y/t*100:.2f}% d'openBlas")
# y = []
# yblas = []
# for l_size in size:
#     total = 0
#     total_blas = 0
#     for _ in range(3):
#         c_code_output = os.popen("./matmul %d"%(l_size)).read()
# 
#         if "ERROR ! MATRIX DIFF !" in c_code_output:
#             print(c_code_output)
#             exit()
# 
#         openblas_time, custom_time = [float(line.split()[-2]) for line in c_code_output.splitlines()]
# 
#         print ("OpenBLAS at size %d \t time %f s"%(l_size, openblas_time))
#         print ("Custom at size %d \t time %f s"%(l_size, custom_time))
#         total += custom_time
#         total_blas += openblas_time
#     y.append(total/3)
#     yblas.append(total_blas/3)
# plt.plot(size,size**3/y, label="Custom")
# plt.plot(size,size**3/yblas, label="OpenBLAS")
# plt.xlabel("Matrix size")
# plt.ylabel("FLOPS")
# plt.legend()
# plt.savefig("v6_moyenne.png")
# plt.show()
#Use this wrapper to create loops over l_size for the different versions of matmul in C and plot the FLOPS curves

# t_custom = []
# t_openblas = []
# for i in range(1,16):
#     os.environ["OMP_NUM_THREADS"] = f"{i}"
#     os.environ["OPENBLAS_NUM_THREADS"] = f"{i}"
# 
#     if os.system("gcc matmul_vect.c -o matmul -O2 -fopenmp -ffast-math -funroll-loops -march=native -lm -lopenblas"): sys.exit("Compilation error")
# 
#     t = 0
#     y = 0
#     for _ in range(3):
#         c_code_output = os.popen("./matmul %d"%(l_size)).read()
#         if "ERROR ! MATRIX DIFF !" in c_code_output:
#             print(c_code_output)
#             exit()
#     # 
# # 
# # 
# # 
# # 
# # 
# # 
# 
#         openblas_time, custom_time = [float(line.split()[-2]) for line in c_code_output.splitlines()]
#     
#         print ("OpenBLAS at size %d and threads %d \t time %f s"%(l_size, i, openblas_time))
#         print ("Custom at size %d \t time %f s"%(l_size, custom_time))
#         t+= custom_time
#         y+= openblas_time
#     t_custom.append(t)
#     t_openblas.append(y)
# 
# plt.plot(range(1,16),3*l_size**3/np.array(t_custom), label="Custom")
# plt.plot(range(1,16),3*l_size**3/np.array(t_openblas), label="OpenBLAS")
# plt.xlabel("Threads")
# plt.ylabel("Flops")
# plt.legend()
# plt.savefig("threads_flops9600.png")
# plt.show()
