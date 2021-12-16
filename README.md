# Math714_final_project_ENO_Burger
Math 714 final project by Ziheng Zhang. The project is about high order ENO finite volume scheme for Burgers' equation. 


**Interpolation file**
The interpolation file only test the ENO interpolation and reconstruction itself, not related to PDEs application. The test function that I use is the smooth sine wave function. I feed the cell averages (exact) into the ENO procedure, and it gives the interpolation value at the cell boundaries. Then it's compared with the true values at those boundaries, then a plot of error is generated in relation to the space discritization h. 

