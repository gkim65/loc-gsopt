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

------------------------------------------------------------------------------------------------------------------------------------------------

# 11/11/2024 Next few objectives: IMPORTANT


Little features (should do first):
- Add in the automatic turn off method when you know it’s all optimized
    - all within 50 meters of each other?
    - get centroid of polygon (and then compute distances from each point and see how it fares)
- starting locations at all most outside locations of the polygon
- gpu parallel processing
    - optimizer line 112 in gsopt
    - mpctx
    - setting up tasks in a specific manner

Other features:
- change between land masses easily when using nelder mead funciton

Things to just start reading over
- current approaches to ground station optimization methods

New experiments
- different satellite cases
    - inclined orbits vs not inclined, do ground staiton locations change
- factors to test on:
    - total data downlink?
    - total number of contacts?
    - minimum average gap time between contacts?
    - satellite constellations, how to best select networks to best optimize for revenue
    - as a GSaAS provider, how to choose best places to past revenue for moving to put gs in some place sthan others
    
- maybe to standardize, choose best location over each continent, then check which continent has best one
    - then iterate over?
    - need to add constraints on how close these stations can be to one another
- compare different points with teleport map?
    - teleports https://www.worldteleport.org/page/TeleportMap3110
    - (comparison between teleports and the potential locations for selection )


- htop
- https://woodstock.stanford.edu/citizen (checking the cpu gpu usage)

------------------------------------------------------------------------------------------------------------------------------------------------
# 11/18/2024 current stuff I've done today and last week

- Kind of implemented the automatic turn off feature, the only thing is that its not converging (tried 10,000m) still not converting >> mabye closer to 100 kms?
- might be prolbem with nelder mead?
    - **I also need to print out (Very quick todo when i get back) print out all of the different converging times so that I can plot that later?**
        - can plot both centroid and distances betwen each point


- finished: gpu parallel processing > completed, tasks currently are just contacts for each point (could we optimize this further?) 
    

TODOS:
- starting locations at all most outside locations of the polygon
    - tried this but im getting into a weird palce where I don't quite get the furthest poinst away from each thing...
    - maybe we should just hand pick these points? rather than compute them every time?
- we should get the best points within the three and return that as our ground station to work with
- for cases where we keep going back and forth, not worth just letting them keep going back and forth
    - convergence cases (not converging further)
- is 100 meters good enough
- running multiple nelder meads and checking best one
    - need to practice running on cluster instead
- don't want to take points that are worse than the new one (need to check if random points are worse)


# jan 20, 2025

- several notes to duncan, rlp .australia for continent doesn;t work
- also i'm not sure if 

# jan 21, 2025

 1093  git status
 1094  git add requirements.txt
 1095  git status
 1096  git add .gitignore
 1097  git status
 1098  git commit -m "trying to make it easier to make virtual environment"
 1099  git pus
 1100  git push
 1101  history
 1102  pip list
 1103  uv pip list


 ****** notes for grace to make requirements.txt file (delete existing requriemetns.txt file before urnning these commands):

```
 pip-compile pyproject.toml --resolver=backtracking
```

 1104  pip-compile pyproject.toml --resolver=backtracking\n
 1105  git status
 1106  git add -A
 1107  git commit -m "more dependencies updates"
 1108  git push

###########################################################################
 1036  python hello.py
 1037  python hello.py
 1038  pip uninstall brahe\n
 1039  pip install git+https://github.com/duncaneddy/brahe.git@master\n
 1040  python hello.py
 1041  cd ..
 1042  deactivate
 1043  git clone https://github.com/gkim65/loc-gsopt.git
 1044  cd loc-gsopt
 1045  python -m venv .venv
 1046  source .venv/bin/activate
 1047  ls
 1048  pip install requirements.txt
 1049  pip install -r requirements.txt
 1050  python hello.py
 1051  python hello.py


# Feb 5, 2025

- Need to check penalty things
    - also the distnaces need to check that
- also need to fix objective function
- testing out different start points nelder mead

# Feb 26, 2025
### Free Select CODE LIST
- need to check for how switching out coordinates gotta implement that at one point
- then also compare with GA
- each experiment set up with hydra
- also set up other objective functions
- set up also weights and biases
    - TODO: put in additional stats per satellite etc for mean, for now just flatten everything 
            - loc-gsopt/src/common/objective_functions.py
            - theres a way to put in plot images
- tmux? try to get this 
    
THINGS I DID:
- did contact times objective function
- added extra things into hydra

NEXT THING: weights and biases set this up!!!!
- then run experiments for some of the base cases
- and try it on the server so it can just run in the background



# March 3, 2025
TODO: Find best initial simplex to start, may need to change based on continents

`loc-gsopt/src/methods/free_select/nelder_mead_scipy.py`

initial_simplex =  np.array([[-30, -60],[-100, -60],[-30, -90]])
initial_simplex =  np.array([[-180, -90],[180, 0],[-180, 90]])

initial_simplex = np.array([
        [-60, -85],  # Point 1 (near the Caribbean)
        [151.2093, -33.8688],  # Point 2 (Sydney, Australia)
        [-153.67891140271288, 55.17076207063536]  # Point 3 (near Alaska)
        ])



THINGS I DID:
- set up most of weights and biases
- figured out how to run on the server
- trying out most of the initial simplex locations
    - tmux session is currently running :D



NEED TO DO:
- Need to run for longer than 1 day!!!
- then run experiments for some of the base cases
