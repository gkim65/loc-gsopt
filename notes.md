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
    - **I also need to print out (Very quick to do when i get back) print out all of the different converging times so that I can plot that later?**
        - can plot both centroid and distances betwen each point


- finished: gpu parallel processing > completed, tasks currently are just contacts for each point (could we optimize this further?) 
    

TO DOS > good for methodology section!:
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
    - TO DO: put in additional stats per satellite etc for mean, for now just flatten everything 
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
TO DO: Find best initial simplex to start, may need to change based on continents

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

# march 27, 2025

was setting up gurobi for ksat experiments, got stopped here: 

https://www.gurobi.com/features/academic-named-user-license/?_gl=1*1gblbgu*_up*MQ..*_ga*Mjk1NjE4NTE0LjE3NDMwOTIyMTY.*_ga_RTTPP25C8N*MTc0MzA5MjIxNi4xLjAuMTc0MzA5MjIxNi4wLjAuNDU2MTA4MjU0 

step 4

# march 28, 2025
need to do comparisons between the three methods, teleports, ksat, and my free select

# march 31, 2025
figured out several things... just writing out several mistakes / things I need to implement

- the biggest thing I need to implement is a 1 to 1 contact constraint (I need something to calculate this right after easily, i'm not computing it right at the moment)
    - Although i can still optimize without needing to do the 1 to 1 contact constraint and it does look like it does alot better in some cases
- the different simplexes also give different answers... i think I could explain this as like a choosing whichever works kinda deal
- trying to also make sure that I'm selecting stations that don't have overlapping comunication cones
- Also, concerns with ground stations that are on islands (kind of sus), not as realistic I guess
- Would be nice to combine duncan's code into ours? but not completely needed I guess
- data downlink is just multiply everything by 1200 MB, but eh its okay
- adding in additional plots into wandb



Notes for vedant:
- line 57 in ILP.py >> try to put in instead into the config file
- line 136 >> get the import in the top of the file again
- maybe worth putting in ilp_model into utils instead?
- I don't think you're doing the 1 contact per sat per ground station thing (its confusing yea def)
- also elevation need to watch out for those when calculating ground station contacts, its in the return_bdm_gs function

# april 1, 2025
- (COMPLETE! But only for Data_downlink) ok we figured out the 1 to 1 contact constraint, might need to see if theres a huge difference just optimizing as it is or optimizing with this constraint too >> at least make sure the outputs give the right thing for data downlink vs gap time (We might need to test more for gap time >>> I'm not sure if thats running correctly)
    - at least I have the function now so that should be able to just give us the best contacts to select for gap times
- we should make a way to save duncan's results into wandb as well >> at least so that all the results are on one place
- save the jsons into wandb too or some other values we can compare and contrast with
- (COMPLETE!) for speedup in nelder mead, save the contacts from each preceding ground station that has been added on already
- need to play around with number of iterations allowed until convergence...

for actual experiments to run, lets start with the following
- data downlink for capella, iceye, maybe flocks (not sure on this one yet)
    - for ground stations numbered 1 - 10
        - for all 3 methods
            - nelder mead run 6 times with 6 different starting configurations
- write out results, and how much they differ

writing sections:

introduction
methods
- nelder mead
    - For nelder mead, we have several things to consider. The output may not be the globally optimal solution, as there are several local minima that the solutions could fall into. But in order to address this, we provide 6 different start simplex configurations in order to ensure we are testing as many methods as possible within the domain. 
experiental setup
- Following the results discussed in Eddy's (insert citaiton here) we run our simulations for a week long period, which should have a good representation of the full scope of contacts a ground station provider would see for various satellite constellations in a year long period. Further discussion on those results is provided in that paper.
- We consider the problem of ground station location optimization of 3 different satellite constellation providers, CAPELLA, ICEYE, and Planet (TBD on that one). These provide a good overview of the types of satellites serviced by the current EO constellation industry, 
results
- several plots:
    - diminishing return of adding in ground stations over time? maybe we can show an optimal number of ground stations to place that can provide good enough coverage for an entire earth
        - may be true for gap time, maybe not for downlink (unless theres just a huge buttload of stations)
    - Showing several good solutions for the same constellation (show that there isn't just one perfect solution), it's more about the distribution of the ground station locations themselves
    - Providing information on benefit of changing out stations over time? idk if its worth showing
        - comparing genetic algorithms vs nelder mead just as it is
    - the affect of putting in more latitude vs longitude (we could definitely show this with the contact duration times)
    - comparing how much better we could get with free select rather than just ksat / teleport locations

# april 7, 2025

- we can find the best set of points to start from from 100? generated random points and select the best three to do our coordinate starts from???
- ok the starting simplex thing isn't really beneficial lol
- but, we do have better working nelder mead... in some cases
i do want to try running the basinhoppin
and also try running with no penalty on going next to other ground statoins, this would help prove the antenna argument


also more plots
- converging times
- TO DO: put in additional stats per satellite etc for mean, for now just flatten everything 
    - loc-gsopt/src/common/objective_functions.py


# april 8 2025
Have been working really hard to try and find a good simplex starter for nelder mead (for that novelty aspect!), to be honest that is still in the works (but might have a breakthrough soon? will let you know)

part of the issues I've been having is that the nelder mead solver kept getting stuck at the poles (which also was why different random seeds had such different answers at times), but I actually figured out a way to get around this by converting the lat long coordinates to a unit sphere so that the poles is better accounted for

And its consistently? (I need to test this more robustly, but so far yes!!) doing better than ksat, but is showing pretty similar results which is really exciting! I think the method definitely is doing better

# april 10, 2025
need to add in the windows plot from vedant's code!

# april 16, 2025
Things I did
- added automatic don't update the txt files
- added seeded
- cleaned up a good amount of todos and other things
- I think the simplex choosing algorithm works, I just need to prove it agressively that it chooses better than random
ok now need todo experiments
several tings I can do: make GA (I think theres a scipy thing for this)
Might just run 