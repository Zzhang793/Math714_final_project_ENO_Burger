# Inviscid Burger's equation: du/dt + u * du/dx = 0, with periodic boundary condition
# Now, using the finite volume instead of finite difference
import numpy as np

# importing the plotting library
import matplotlib.pyplot as plt

# Define the initial periodic condition
def f(x):
    return 3/2 + np.sin(np.pi*2*x)

# Step function initial condition
# def f(x):
#     # Define a step function, jump at x=0.5
#     Nx = np.size(x)
#     output = np.zeros(Nx)
#     for i in range(Nx):
#         if x[i]>0.5:
#             output[i]=2
#         else:
#             output[i]=1
#     return output

# Cell average of the step function
# def g(x_i,h):
#     Nx = np.size(x_i)
#     output = np.zeros(Nx)
#     x_i = np.append(x_i, 1)
#     for i in range(Nx):
#         if x_i[i]<=0.5 and x_i[i+1]<0.5:
#             output[i] = 1
#         elif x_i[i]>0.5 and x_i[i+1]>=0.5:
#             output[i] = 2
#         elif x_i[i]<=0.5 and x_i[i+1]>=0.5:
#             output[i] = ((0.5-x_i[i])*1 + (x_i[i+1]-0.5)*2)/h
#         else:
#             raise("Something is wrong")
#     return output

def g(x_i,h):   # Average of f(x) over each cell
    Nx = np.size(x_i)
    u_bar = np.zeros(Nx)
    x_i = np.append(x_i,0)
    for i in range(Nx):
        u_bar[i] = 3/2 + 1/(2*np.pi*h)*(np.cos(2*np.pi*x_i[i])-np.cos(2*np.pi*x_i[i+1]))
    return u_bar


# Input U is the value on the grid, K is the order to which we want to compute
def Newton_divided_difference(U,Nx,h,K):
    Table = np.zeros(shape=(Nx,K))
    Table[:,0] = U[:]

    for k in range(0,K-1):
        for j in range(Nx-k-1):
            Table[j,k+1] = (Table[j+1,k] - Table[j,k])/((k+1)*h)
    return Table

# Current version for 4th order, k=4 or 4 stencils point for each interpolation polynomial
def ENO_interpolation_4th_order(pre_U,Nx,h,k):
    if k!=4:
        raise Warning("This ENO works for k=4")

    # Define the helper vector with two end extended as a result of the periodicity.
    UU = np.zeros(Nx + 2 * k)
    UU[k:Nx + k] = pre_U[:]
    UU[0:k] = pre_U[Nx - k:Nx]
    UU[Nx + k:Nx + k * 2] = pre_U[0:k]

    # Then use ENO procedure to find the value at each interface

    # First, find the difference table
    difference_table = np.abs(Newton_divided_difference(UU, Nx + 2 * k, h, 4))

    v_plus = np.zeros(Nx)
    v_minus = np.zeros(Nx)
    # Then use the difference table to determine stencil we use for each interpolation point

    # Computing v_plus, left side extension is used
    for i in range(Nx):  # i=0 corresponds to the interface at x=0
        left_end = i - 1  # this is going to be -1 if i=0
        right_end = i - 1  # The initial stencil always include the i-1 cell and i+1 cell

        for ll in range(1, k):
            if difference_table[left_end + k - 1, ll] < difference_table[left_end + k, ll]:
                left_end = left_end - 1
                # print("Choosing left")
            else:
                right_end = right_end + 1
                # print("Choosing right")

        # Now, we have 4 stencils, compute the value at the interface value
        if left_end == i - 4:
            aa = np.array([-1 / 4, 13 / 12, -23 / 12, 25 / 12])
        elif left_end == i - 3:
            aa = np.array([1 / 12, -5 / 12, 13 / 12, 1 / 4])
        elif left_end == i - 2:
            aa = np.array([-1 / 12, 7 / 12, 7 / 12, -1 / 12])
        elif left_end == i - 1:
            aa = np.array([1 / 4, 13 / 12, -5 / 12, 1 / 12])
        elif left_end == i:
            aa = np.array([25 / 12, -23 / 12, 13 / 12, -1 / 4])
            # print("This should never happen")
        else:
            Warning("Something is wrong")
        v_plus[i] = np.dot(aa, UU[left_end + k:right_end + k + 1])

        # Then try to compute the v_minus, the initial stencil is now on the right
        left_end = i
        right_end = i

        for ll in range(1, k):
            if difference_table[left_end + 3, ll] < difference_table[left_end + 4, ll]:
                left_end = left_end - 1
                # print("Choosing left")
            else:
                right_end = right_end + 1
                # print("Choosing right")

                # Now, we have 4 stencils, compute the value at the interface value
        if left_end == i - 4:
            aa = np.array([-1 / 4, 13 / 12, -23 / 12, 25 / 12])
            # print("This should never happen")
        elif left_end == i - 3:
            aa = np.array([1 / 12, -5 / 12, 13 / 12, 1 / 4])
        elif left_end == i - 2:
            aa = np.array([-1 / 12, 7 / 12, 7 / 12, -1 / 12])
        elif left_end == i - 1:
            aa = np.array([1 / 4, 13 / 12, -5 / 12, 1 / 12])
        elif left_end == i:
            aa = np.array([25 / 12, -23 / 12, 13 / 12, -1 / 4])
        else:
            Warning("Something is wrong")
        v_minus[i] = np.dot(aa, UU[left_end + k:right_end + k + 1])

    return v_plus, v_minus

def ENO_interpolation_3rd_order(pre_U,Nx,h,k):

    if k!=3:
        raise Warning("This ENO works for k=3")
    # Define the helper vector with two end extended as a result of the periodicity.
    UU = np.zeros(Nx + 2 * k)
    UU[k:Nx + k] = pre_U[:]
    UU[0:k] = pre_U[Nx - k:Nx]
    UU[Nx + k:Nx + k * 2] = pre_U[0:k]

    # Then use ENO procedure to find the value at each interface

    # First, find the difference table
    difference_table = np.abs(Newton_divided_difference(UU, Nx + 2 * k, h, 4))

    v_plus = np.zeros(Nx)
    v_minus = np.zeros(Nx)
    # Then use the difference table to determine stencil we use for each interpolation point

    # Computing v_plus, left side extension is used
    for i in range(Nx):  # i=0 corresponds to the interface at x=0
        left_end = i - 1  # this is going to be -1 if i=0
        right_end = i - 1  # The initial stencil always include the i-1 cell and i+1 cell

        for ll in range(1, k):
            if difference_table[left_end + k - 1, ll] < difference_table[left_end + k, ll]:
                left_end = left_end - 1
                # print("Choosing left")
            else:
                right_end = right_end + 1
                # print("Choosing right")

        # Now, we have 3 stencils, compute the value at the interface value
        if left_end == i - 3:
            aa = np.array([1/3, -7/6, 11/6])
        elif left_end == i - 2:
            aa = np.array([-1/6, 5/6, 1/3])
        elif left_end == i - 1:
            aa = np.array([1/3, 5/6, -1/6])
        elif left_end == i:
            aa = np.array([11/6, -7/6, 1/3])
            print("this should never be happening")
        else:
            Warning("Something is wrong")
        v_plus[i] = np.dot(aa, UU[left_end + k:right_end + k + 1])

        # Then try to compute the v_minus, the initial stencil is now on the right
        left_end = i
        right_end = i

        for ll in range(1, k):
            if difference_table[left_end + 3, ll] < difference_table[left_end + 4, ll]:
                left_end = left_end - 1
                # print("Choosing left")
            else:
                right_end = right_end + 1
                # print("Choosing right")

                # Now, we have 4 stencils, compute the value at the interface value
        if left_end == i - 3:
            aa = np.array([1 / 3, -7 / 6, 11 / 6])
            print("this should never be happening")
        elif left_end == i - 2:
            aa = np.array([-1 / 6, 5 / 6, 1 / 3])
        elif left_end == i - 1:
            aa = np.array([1 / 3, 5 / 6, -1 / 6])
        elif left_end == i:
            aa = np.array([11 / 6, -7 / 6, 1 / 3])
        else:
            Warning("Something is wrong")
        v_minus[i] = np.dot(aa, UU[left_end + k:right_end + k + 1])

    return v_plus, v_minus

# Now, check that my ENO interpolation is of 4th order accuracy
error_max = []
error_l1 = []
NN = []
for i in range(4,11):
    Nx = 2**i
    h = 1 / Nx
    x_i = np.linspace(0, 1 - h, Nx)
    pre_U = g(x_i, h)
    v_plus, v_minus = ENO_interpolation_4th_order(pre_U, Nx, h, 4)

    # fig, (ax1, ax2) = plt.subplots(1, 2)
    # fig.set_figheight(2.5)
    # fig.set_figwidth(8)
    # # ax1.plot(x_i,f(x_i),linewidth=1)
    # ax1.plot(x_i+0.5*h,pre_U,'r-o')
    # ax1.scatter(x_i, v_plus)
    # ax1.set_title("u_minus")
    # ax1.legend(["cell averages","u^(-)"])
    #
    # ax2.plot(x_i+0.5*h,pre_U,'r-o')
    # ax2.scatter(x_i,v_minus)
    # ax2.set_title("u_plus")
    # ax2.legend(["cell averages", "u^(+)"])
    # plt.show()

    ref = f(x_i)  # The real value at those cell boundary points
    max_error = np.max(np.abs(ref - v_plus))
    NN.append(Nx)
    error_max.append(max_error)

plt.figure(figsize=(5,3.5))
plt.plot(NN,error_max,'b-o')
# plt.plot(NN,[N**(-4) for N in NN])
plt.plot(NN,[N**(-4) for N in NN])
plt.xscale("log")
plt.yscale("log")
plt.title("(b) Error plot for 4th order ENO")
plt.xlabel("N")
plt.ylabel("Error")
plt.legend(["Max_error","N^(-4)"])
plt.show()