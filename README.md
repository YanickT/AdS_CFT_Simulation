# AdS/CFT-correspondence (simulation for Bachelor Thesis)

## Preface
This is the repository to my Bachelor Thesis in physics. 
The purpose of this thesis is to compare the influcence of different
discretizations of the Anti-de-Sitter-space (AdS-space).  
Therefore, two discritziations were used. The first one can be found in the 
article "Topological Complexity in AdS3/CFT2".  
The secound one is developed in the thesis.
A special thanks to one of the authors for providing the C++ for the square geometry
discretization. The code was translated into pyhton.
The mathematical and theoretical basics for the constant spin-density discretizaiton
was done in my thesis and thus will not be explained in detail within this repository.

## Short overview
The AdS/CFT-correspondence is a physical tool which connects gravitational theories in 
a higher dimensional AdS-space to conformal field theories (CFT).
Due to the holographic principle the information, which in general is proportial to entropy,
is stored on the boundary of gravitational objects. 
By adding a black-hole to the AdS-space the entanglement entropy in CFT can be calculated.
This is advantagous since only in some few cases it is possible to calculate it directly in the framework
of CFTS.
Furthermore, is it possible to approximate the entanglement entropy using Ising models.
Some physical quantities, like complexity, are better understood in the context of Ising models than CFTs.
This was used in the article mentioned above. Unfortuntly, leads the used quantization of the AdS-space
to deviations. 
In the case of a 2D AdS timeslice does the horrizon, carring the information in the AdS-space, resemble the 
geodesic. Therefore, the results of these simulations can be compared to it. 

## Numba vs C-API
The first simulation was done using Pyhtons & Numpys C-API. 
Due to the massive performace boost using the just-in-time compiler numba the API-versions were not developed any further and are not up-to-date.
In the thesis the results of the numba-versions were used.  
