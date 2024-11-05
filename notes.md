Using Nelder Mead:

- https://en.wikipedia.org/wiki/Nelder%E2%80%93Mead_method
- https://docs.scipy.org/doc/scipy/reference/optimize.minimize-neldermead.html#optimize-minimize-neldermead

General optimization scipy

- https://scipy.github.io/devdocs/tutorial/optimize.html

Determining if coordinates are on land or ocean:

- https://stackoverflow.com/questions/52761380/determining-if-a-place-given-coordinates-is-on-land-or-on-ocean

nelder mead by hand:

- https://brandewinder.com/2022/03/31/breaking-down-Nelder-Mead/

Getting uniformly distributed numbers on a sphere

- https://corysimon.github.io/articles/uniformdistn-on-sphere/

Sampling technique >> polygon list of countries
- take out list of countries that we don't want to >> 

communications cones for ground stations 
- means altitudes of the constellations
- how much area is the station covering 

optimization questions: 
- centroid of every single country >> optimize over the best over all countries
- extend up to the boundary of the polygon

different test cases of satellites
- capella has the inclined orbits 

heuristics of each country
- have some heuristics on 
- only place on regions that rae outside of communications cones >> update polygon at each placement

Things I need to add in 10/28/24
- the communication cones
- implement nelder mead if its not on land
- randomized points with duncan's function


