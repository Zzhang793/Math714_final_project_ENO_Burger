# Inviscid Burger's equation: du/dt + u * du/dx = 0, with periodic boundary condition
# Now, using the finite volume instead of finite difference
import numpy as np

# importing the plotting library
import matplotlib.pyplot as plt

# Define the initial periodic condition
def f(x):
    # Define a step function, jump at x=0.5
    # Nx = np.size(x)
    # output = np.zeros(Nx)
    # for i in range(Nx):
    #     if x[i]>0.5:
    #         output[i]=2
    #     else:
    #         output[i]=1
    # return output
    return 3/2 + np.sin(np.pi*2*x)

def g(x_i,h):
    Nx = np.size(x_i)
    u_bar = np.zeros(Nx)
    x_i = np.append(x_i,0)
    for i in range(Nx):
        u_bar[i] = 3/2 + 1/(2*np.pi*h)*(np.cos(2*np.pi*x_i[i])-np.cos(2*np.pi*x_i[i+1]))
    return u_bar

# Upwind, but also taking into account of the transonic rarefaction case
def transonic_rarefaction_finite_volume_solver(Nx,NT,dt):
    # Grid Spacing
    print("Solving upwind finite volume Euler forward scheme with Nx="+str(Nx)+" dt="+str(dt))
    h = 1. / Nx

    if h/dt > 2.5:
        Warning("CFL condition not satisfied, please use smaller dt")

    # Cell boundaries
    x_i = np.linspace(0,1-h,Nx)

    # Set the U at t=0 to the initial condition
    # U the cell average, where the first one is the cell average over [0,h]
    # Now the initial cell average simply takes the center value
    # pre_U = f(x_i+0.5*h)
    pre_U = g(x_i,h)
    print("Total mass at t=0 is "+str(np.sum(pre_U)*h))
    curr_U = np.zeros(Nx)

    for tt in range(NT):
        # Helper vector
        UU = np.zeros(Nx+2)
        UU[1:Nx+1] = pre_U[:]
        UU[0] = pre_U[Nx-1]
        UU[Nx+1] = pre_U[0]

        for i in range(1, Nx+1):
            a = UU[i-1]
            b = UU[i]
            c = UU[i+1]
            # Now, compute the accumulated effect (flux from both sides)
            change = 0

            # The following in an entropy fix
            if a<0 and b>0:
                change = change - (b ** 2) / 2 * dt / h
            elif a + b >= 0:  # Moving to the right, then compute the change from left cell
                change = change + (a + b) / 2 * dt * (a - b) / h
            if b<0 and c>0:
                change = change + (a**2)/2*dt/h
            elif b + c < 0:  # Moving to the left, use right and middle cell
                change = change + (b + c) / 2 * dt * (b - c) / h
            curr_U[i-1] = pre_U[i-1] + change

        # Update the pre to be current for next iteration
        pre_U[:] = curr_U[:]
    print("Total mass at final time="+str(np.sum(pre_U)*h))
    return pre_U

def Gudonov_flux(a,b):
    if a<=b:
        if a<=0 and b>=0:
            return 0
        elif a<=0 and b<=0:
            return (b**2)/2
        else:
            return (a**2)/2
    else:
        if np.abs(a)>=np.abs(b):
            return (a**2)/2
        else:
            return (b**2)/2
    # return (a**2)/2

# Input a average of left cell, b:value at x_1 from left,
# c value at x_1 from right, d: average of right cell
def linear_reconstruction_flux(a,b,c,d,h,dt):
    if a>0 and d>0:
        alpha = (b-a)/(h/2)
    else:
        print("Something is wrong")
    return 1/2*(b**2)/(alpha*dt+1)

# Input U is the value on the grid, K is the order to which we want to compute
def Newton_divided_difference(U,Nx,h,K):
    Table = np.zeros(shape=(Nx,K))
    Table[:,0] = U[:]

    for k in range(0,K-1):
        for j in range(Nx-k-1):
            Table[j,k+1] = (Table[j+1,k] - Table[j,k])/((k+1)*h)
    return Table

def upwind_finite_difference_solver(Nx,NT,dt):
    #Grid Spacing
    print("Solving upwind FD Euler foward with Nx="+str(Nx)+" dt="+str(dt))
    h = 1. / Nx

    if h/dt > 2.5:
        Warning("CFL condition not satisfied, please use smaller dt")

    # Grid points for the unknowns, [0,h,2h,...,1-h]
    x_i = np.linspace(0,1-h,Nx)
    # Set the U at t=0 to the initial condition
    pre_U = f(x_i)
    curr_U = np.zeros(Nx)

    for tt in range(NT-1):
        # Helper vector with extended ends, because our scheme is periodic
        UU = np.zeros(Nx + 2)
        UU[1:Nx + 1] = pre_U[:]
        UU[0] = pre_U[Nx - 1]
        UU[Nx + 1] = pre_U[0]

        for i in range(1,Nx+1):
            # 1st order upwind scheme
            if pre_U[i-1]>0:
                dudx = (UU[i] - UU[i-1])/h
            else:
                dudx = (UU[i+1] - UU[i])/h
            curr_U[i-1] = pre_U[i-1] - dt*pre_U[i-1]*dudx

        # Update the pre to be current for next iteration
        pre_U[:] = curr_U[:]

    return pre_U

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
    difference_table = np.abs(Newton_divided_difference(UU, Nx + 2 * k, h, 3))

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
            if difference_table[left_end + k - 1, ll] < difference_table[left_end + k, ll]:
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
            if difference_table[left_end + k - 1, ll] < difference_table[left_end + k, ll]:
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

def ENO_finite_volume_solver_higher_order_ENO(Nx,NT,dt):
    # Grid Spacing
    print("Solving ENO(3rd) order with linear reconstruction forward Euler scheme with Nx="+str(Nx)+" dt="+str(dt))
    h = 1. / Nx

    if h/dt > 5:
        Warning("CFL condition not satisfied, please use smaller dt")

    # Cell boundaries
    x_i = np.linspace(0,1-h,Nx)

    # Set the U at t=0 to the initial condition
    # U the cell average, where the first one is the cell average over [0,h]
    # Now the initial cell average simply takes the center value
    # pre_U = f(x_i+0.5*h)
    # Instead of naive average, use integration to determine the average
    pre_U = g(x_i,h)
    print("Total mass at t=0 is "+str(np.sum(pre_U)*h))
    curr_U = np.zeros(Nx)

    k=5   # k=4 is needed to get 4 stencils for each point
    for tt in range(NT):
        v_plus, v_minus = ENO_interpolation_4th_order(pre_U,Nx,h,4)

        # Compute the Godonov flux, then use flux to update each cell average
        flux = np.zeros(Nx)
        for i in range(Nx):
            if i==0:
                a =pre_U[Nx-1]
            else:
                a = pre_U[i-1]
            d = pre_U[i]
            b = v_plus[i]
            c = v_minus[i]
            flux[i] = Gudonov_flux(b,c)
            #flux[i] = linear_reconstruction_flux(a, b, c, d, h, dt)
        flux = np.append(flux,flux[0])
        for i in range(Nx):
            curr_U[i] = pre_U[i] + flux[i]*dt/h - flux[i+1]*dt/h

        # Update the pre to be current for next iteration
        pre_U[:] = curr_U[:]

    print("Total mass at final time="+str(np.sum(pre_U)*h))

    return pre_U

# ENO_interpolation_5th_order(pre_U,Nx,h,k):
def ENO_interpolation_5th_order(pre_U,Nx,h,k):
    if k!=5:
        raise Warning("This ENO works for k=5")

    # Define the helper vector with two end extended as a result of the periodicity.
    UU = np.zeros(Nx + 2 * k)
    UU[k:Nx + k] = pre_U[:]
    UU[0:k] = pre_U[Nx - k:Nx]
    UU[Nx + k:Nx + k * 2] = pre_U[0:k]

    # Then use ENO procedure to find the value at each interface

    # First, find the difference table
    difference_table = np.abs(Newton_divided_difference(UU, Nx + 2 * k, h, k))

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

        # Now, we have 5 stencils, compute the value at the interface value
        if left_end == i - 5:
            aa = np.array([1/5,-21/20,137/60,-163/60,137/60])
        elif left_end == i - 4:
            aa = np.array([-1/20,17/60,-43/60,77/60,1/5])
        elif left_end == i - 3:
            aa = np.array([1/30,-13/60,47/60,9/20,-1/20])
        elif left_end == i - 2:
            aa = np.array([-1/20,9/20,47/60,-13/60,1/30])
        elif left_end == i - 1:
            aa = np.array([1/5,77/60,-43/60,17/60,-1/20])
        elif left_end == i:
            aa = np.array([137/60,-163/60,137/60,-21/20,1/5])
            raise Warning("Something is wrong")
        else:
            raise Warning("Something is wrong")
        v_plus[i] = np.dot(aa, UU[left_end + k:right_end + k + 1])

        # Then try to compute the v_minus, the initial stencil is now on the right
        left_end = i
        right_end = i

        for ll in range(1, k):
            if difference_table[left_end + k - 1, ll] < difference_table[left_end + k, ll]:
                left_end = left_end - 1
                # print("Choosing left")
            else:
                right_end = right_end + 1
                # print("Choosing right")

        # Now, we have 5 stencils, compute the value at the interface value
        if left_end == i - 5:
            aa = np.array([1 / 5, -21 / 20, 137 / 60, -163 / 60, 137 / 60])
            raise Warning("Something is wrong")
        elif left_end == i - 4:
            aa = np.array([-1 / 20, 17 / 60, -43 / 60, 77 / 60, 1 / 5])
        elif left_end == i - 3:
            aa = np.array([1 / 30, -13 / 60, 47 / 60, 9 / 20, -1 / 20])
        elif left_end == i - 2:
            aa = np.array([-1 / 20, 9 / 20, 47 / 60, -13 / 60, 1 / 30])
        elif left_end == i - 1:
            aa = np.array([1 / 5, 77 / 60, -43 / 60, 17 / 60, -1 / 20])
        elif left_end == i:
            aa = np.array([137 / 60, -163 / 60, 137 / 60, -21 / 20, 1 / 5])
        else:
            raise Warning("Something is wrong")
        v_minus[i] = np.dot(aa, UU[left_end + k:right_end + k + 1])

    return v_plus, v_minus

def ENO_finite_volume_solver_higher_order_ENO_2nd_Runge_Kutta(Nx,NT,dt):  # trying new things
    # Grid Spacing
    print("Solving 4th order ENO (linear reconstruction) 2nd order Runge Kutta with Nx="+str(Nx)+" dt="+str(dt))
    h = 1. / Nx

    if h/dt > 5:
        Warning("CFL condition not satisfied, please use smaller dt")

    # Cell boundaries
    x_i = np.linspace(0,1-h,Nx)

    # Set the U at t=0 to the initial condition
    # U the cell average, where the first one is the cell average over [0,h]
    # Instead of naive average, use integration to determine the averages
    pre_U = g(x_i,h)
    print("Total mass at t=0 is "+str(np.sum(pre_U)*h))
    U_1 = np.zeros(Nx)
    curr_U = np.zeros(Nx)

    k=3  # k=4 is needed to get 4 stencils for each point
    for tt in range(NT):
        # First, determine the middle step
        v_plus, v_minus = ENO_interpolation_3rd_order(pre_U,Nx,h,k)

        # Compute the Godonov flux, then use flux to update each cell average
        flux = np.zeros(Nx)
        for i in range(Nx):
            if i==0:
                a =pre_U[Nx-1]
            else:
                a = pre_U[i-1]
            d = pre_U[i]
            b = v_plus[i]
            c = v_minus[i]
            flux[i] = Gudonov_flux(b,c)
            # flux[i] = linear_reconstruction_flux(a, b, c, d, h, dt)
        flux = np.append(flux,flux[0])
        for i in range(Nx):
            U_1[i] = pre_U[i] + flux[i]*dt/h - flux[i+1]*dt/h

        # Now, use half_U and pre_U to determine the curr_U
        v_plus, v_minus = ENO_interpolation_3rd_order(U_1, Nx, h, k)
        flux = np.zeros(Nx)
        for i in range(Nx):
            if i == 0:
                a = U_1[Nx - 1]
            else:
                a = U_1[i - 1]
            d = U_1[i]
            b = v_plus[i]
            c = v_minus[i]
            flux[i] = Gudonov_flux(b,c)
            # flux[i] = linear_reconstruction_flux(a, b, c, d, h, dt)
        flux = np.append(flux, flux[0])

        for i in range(Nx):
            curr_U[i] = (pre_U[i] +U_1[i] + flux[i]*dt/h-flux[i+1]*dt/h)/2

        # Update the pre to be current for next iteration
        pre_U[:] = curr_U[:]

    print("Total mass at final time="+str(np.sum(pre_U)*h))

    return pre_U

# The following setup should be used in my report, to demonstrate that this is 5th order in space
# level_ref = 14
# Nx_ref = 2**level_ref   # 2**10 = 1024, 2**14 = 4096
# T =  0.0001
# dt_ref = 0.00001
# h = 1/Nx_ref
# # u_ref= ENO_finite_volume_solver_higher_order_ENO(Nx_ref,int(T/dt_ref),dt_ref)

# Currently trying the following method
level_ref = 8
Nx_ref = 2**level_ref   # 2**10 = 1024, 2**14 = 4096
T =  0.26
dt_ref = 0.0001
h = 1/Nx_ref
u_ref= ENO_finite_volume_solver_higher_order_ENO(Nx_ref,int(T/dt_ref),dt_ref)

### saving the reference solution
# with open('ENO_FV_u_ref.npy', 'wb') as ff:
#     np.save(ff,u_ref)
# Load the reference solution directly
# with open('ENO_FV_u_ref.npy', 'rb') as ff:
# 	u_ref = np.load(ff)
#
# with open('ENO_FV_u_ref_flux.npy', 'wb') as ff:
#     np.save(ff,u_ref)
# with open('ENO_FV_u_ref_flux.npy', 'rb') as ff:
# 	u_ref = np.load(ff)


## Plot the v and compare it with u
x_i = np.linspace(0+0.5*h,1-0.5*h,Nx_ref)
plt.plot(x_i,f(x_i))
plt.plot(x_i,u_ref)
# plt.scatter(x_i,u_ref,s=2)
plt.title("Burgers equation solution at t="+str(T))
# x_i = np.linspace(0,1-h,Nx)
# plt.scatter(x_i,v_minus,s=1)
# plt.scatter(x_i,v_plus,s=1)
plt.legend(["initial condition","numerical solution"])
plt.show()

##############################################################

# we will save the different solution in here
U = []
# Save the N list
NN = []

num_levels = 6   # number of consecutive levels
start_level = 4  # The smallest N will be 2^start_level

# Run the solver for each different level (different N)
for level in range(start_level,start_level+num_levels,1):
    # Computing the number of points to make the grids properly nested
    Nx = 2**(level)

    # We also choose the dt proportional to 1/Nx
    # dt = dt_ref*(2**(level_ref-level))
    dt = dt_ref

    u_h_level = ENO_finite_volume_solver_higher_order_ENO(Nx,int(T/dt),dt)

    # Saving the solution and the grid
    NN.append(Nx)
    U.append(u_h_level)

# Store the error for each level, current using the l2 error
err_h_max = []
err_h_l1 = []
print("Evaluating the error")

for level, (u_h, Nx) in enumerate(zip(U, NN)):
    # computing the down sampling factor
    sample = (u_ref.shape[0]) // (u_h.shape[0])
    # Averaging the u_ref to the coarse grid
    u_ref_average = np.zeros(u_h.shape[0])
    for i in range(u_h.shape[0]):
        u_ref_average[i] = np.average(u_ref[i*sample:(i+1)*sample])
    error = u_h - u_ref_average
    # error = u_h - u_ref[::sample]
    # computing the error in l^2 norm, normalized so it does not scale with Nx
    err_l1 = np.linalg.norm(error,ord=1)/np.size(error)
    # Compute the error in max norm
    err_max = np.linalg.norm(error,np.inf)

    err_h_l1.append(err_l1)
    err_h_max.append(err_max)

# plot the error
plt.figure(2)
plt.plot(NN, err_h_max,'b-o')
plt.plot(NN,err_h_l1,'r-o')
plt.plot(NN,[N**(-3) for N in NN],'k-o')
plt.plot(NN,[N**(-4) for N in NN],'m-o')
plt.xscale("log")
plt.yscale("log")
plt.title("(a) Error for 4th order ENO FV scheme")
plt.xlabel("N")
plt.ylabel("Error")
#plt.legend(["Max_error","O(1/N)","10N^-2"])
plt.legend(["Max_error","L1_error/N","N^(-3)","N^(-4)"])
#
plt.show()

# Add the reference solution to our list
NN.append(Nx_ref)
U.append(u_ref)

# Plot the solutions together in the same plot
plt.figure(3)
for level, (u_h, Nx) in enumerate(zip(U, NN)):
    x_i = np.linspace(0.5/Nx,1-0.5/Nx,Nx)
    plt.plot(x_i,u_h)
plt.title("Figure 3: solution under different grid size for problem A(b)")
plt.xlabel("x")
plt.ylabel("U")
legend = map(str, NN)
# plt.legend(legend)
plt.legend(["Nx=32","Nx=64","Nx=128","Nx=256","Nx=512","Nx="+str(Nx_ref)])
plt.show()

