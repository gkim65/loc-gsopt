# loc-gsopt

Ground station location optimizer. Currently in development.

Example optimizer for nelder mead:
![](https://github.com/gkim65/loc-gsopt/blob/main/gifs/example_figure.gif)


## Instructions for installing dependencies and cloning this repository

First, clone the repository on the folder you'd like to run this in.

```
git clone https://github.com/gkim65/loc-gsopt.git
```

You can then `cd` into the folder created from this command, `loc-gsopt`

```
cd loc-gsopt
```

And create a virtual environment to download all of your dependencies. If you already have a package manager, thats great! If not, I put (at least for a Mac) how to make a simple virtual environment using `venv`. The second line just activates the environment for us to use.

```
python -m venv .venv
source .venv/bin/activate
```

Finally, just run this command to install all the required dependencies in your virtual environment.

```
pip install -r requirements.txt
```

### If using `uv`:

Create a new virtual environment with uv on a Mac inside the git folder, and install dependencies

```
uv venv
source .venv/bin/activate 
uv sync
```




## Using this Repository

Using the `hydra` config file manager, you can run files by running:

```
python src/main.py
```

You can run sweeps/perform parameter runs

```
python src/main.py --multirun problem.sat_num=1,5,10
```

or just change the parameters directly in `config\config.yaml`. 

*NOTE: Example folder `ipynb` files, still in development* (They are runnable, but not fully cleaned up)

