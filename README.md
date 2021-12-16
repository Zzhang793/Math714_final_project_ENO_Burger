# Math714_final_project_ENO_Burger
Math 714 final project by Ziheng Zhang. The project is about high order ENO finite volume scheme for Burgers' equation. 


**ENO_Interpolation.py**
This python file only tests the ENO interpolation and reconstruction itself, not related to PDEs application. The test function that I use is the smooth sine wave function. I feed the cell averages (exact) into the ENO procedure, and it gives the interpolation value at the cell boundaries. Then it's compared with the true values at those boundaries, then a plot of error is generated in relation to the space discretization h. 

**ENO_Burgers.py**
This python file tests the high-order ENO finite volume scheme on Burgers' equation. Firstly, Euler forward with only a tiny time step is used. The reference solution is computed with a highly refined grid. Error as a function of h is plotted. Secondly, a final time of 0.26 is used to see what happens after shock form. 

How to run this code: hit run, and then every time a figure show up, close the figure will continue the program, there are 3 parts of the program that generate 3 different figures. 

The other .npy files are for storage of the reference solution computed on a highly refined grid, which might take a lot of time to run.
