# CUDA based interactive simulation of Ginzburg-Landau equation with parametric forcing 
by Dr. Kjartan Pierre Emilsson

![Cover](./images/cover.jpg  "cover image")
# Background

In 1994 I finished my Phd thesis under the supervision of professor Dr. Pierre Coullet from the University of Nice - Sophia Antipolis. The title of the thesis was "Strong Resonances in a Field of Oscillators and Bifurcation of Defects" and its subject was to investigate the patterns emerging in the two-dimensional Ginzburg-Landau equation in the presence of parametric forcing. 

![Equation](./images/equation.png  "equation")

There are two subjects in this thesis. In the first part, a qualitative method to 
classify and predict the structure of defects in reaction-diffusion systems is
introduced. This qualitative approach makes it easier to analyze the behavior
of defects in complex systems. It also gives us information about the inner 
structure of the defect, and from that point of view, it makes it possible to 
approach the concept of defect bifurcation in a novel manner. In the second 
part, we study the normal form governing the evolution of a spatially extended 
homogeneous temporal instability, in the presence of a temporal forcing. This 
is equivalent to studying strong resonances of a field of nonlinear oscillators. 
A detailed analysis of the phase space of this normal form reveals a rich 
dynamical structure, which gives rise to a variety of spatial structures. These 
include excitable pulses, excitable spirals, fronts and spatially periodic 
structures. These structures are studied and their possible bifurcations are 
analyzed from a qualitative point of view.

Link to [thesis](https://raw.githubusercontent.com/lekjart/ginzburg-landau/main/Thesis/GinzburgLandauParametric_KjartanEmilsson_Phd_Thesis.pdf)

# Simulation

At the time the simulations for this thesis were performed on a [Connection Machine](https://en.wikipedia.org/wiki/Connection_Machine) supercomputer from Thinking Machine and written in C*, the parallel extension to C, but 30 years later I rewrote it to run on CUDA based platforms so it runs very well on any Windows PC with a decent Nvidia GPU. Here you will find both a PDF version of the original thesis as well as the CUDA source code for the simulation.

Link to github [repo](https://github.com/lekjart/ginzburg-landau/)

# Installation and pre-requisites
First you will need to install Visual Studio from Microsoft. For example [Microsoft Community Edition](https://visualstudio.microsoft.com/vs/community/). 

After that you will need to install the [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) from Nvidia.

Go to github and download the [CUDA samples](https://github.com/NVIDIA/cuda-samples) and install it somewhere on your computer.

You should then be able to open the `ThinkingMachine.sln` solution file and then go to `Project - Properties - VC++ Directories`. Edit the `Include Directories` and `Library Directories` to point to where you installed the samples. Do this for both `Release` and `Debug` configurations. You should then be able to rebuild the solution and launch the program.

# Program structure
The program is really simple. It uses [Glut](https://freeglut.sourceforge.net/) to create a simple UI to dynamically change simulation parameters and a window to manage the display of a bitmap that is drawn into by the GPU and simulated by CUDA. All this interface code is in the `Master.cpp` file. All the CUDA specific code is in the `Solver.cu` and `Solver.h` files.

The flow of the CUDA code is the following:
* `InitSolver()` allocates GPU memory buffer that will contain the real and imaginary fields A(t) = (U(t),V(t)) with the resolution `FIELDSIZEX` and `FIELDSIZEY`. It creates one for the A(t0+dt) and another one for A(t0) (new and old).
* It also allocates a buffer to calculate the bitmap from the field value.
* `ResetField()` resets the field to zero with an optional noise overlay controlled by the noise parameter.
* The function `DoStuff()` is then called periodically from the main UI and essentially advances the simulation by calling the `Advance<< >>` function, which is the proper parallel processing and then retrieves the bitmap reflecting current state using the `FillBuff<< >>` function.
  * It optionally takes a `count` parameter that specifies how many iterations it should calculate during each advance. This allow to run the simulation much faster as transfer of bitmap from GPU to PC does not have to be done for each iteration, which is a slow operation. On a Nvidia 3070 GPU this allow to do 300 iterations a second of a 2496x3520 field
 
The CUDA simulation code itself follows standard CUDA programming. It basically subdivides the field into 78x110 blocks of size 32x32, where each block is processed in parallel. Nearest neighbors are either obtained locally within a block (fast) or goes through global memory when straddling block boundaries (slower).

# Running the program
Buid the `Release` version and then run it without debugging (Ctrl-F5). You will see two windows appear: one dialog window with a collection of parameters, choice of field component to visualize, choice of palette (which are can be reloaded from text file if you change them) and a `Reset` button. The other window is the actual simulation window. The initial state of the field is zero with a slight noise.  If you push the `Reset` button, the field will be initialized to zero (possibly adding some noise if the noise parameter is non-zero). Whenever you change a parameter in the parameter list, the change is applied immediately. Be careful in changing the `dx` or `dt` parameter, as extreme values will ruin the simulation.

Clicking with left mouse button anywhere on the field will add a zero there (assuming the field is non-zero). Right clicking will save the current bitmap in a filed called `tmp.bmp`.

Color palettes are done using a lookup table that is parametrized by a 4 anchor Bezier curve in RGB space. Each palette is defined by a name and 4 rgb points in the file `palettes.txt`.
