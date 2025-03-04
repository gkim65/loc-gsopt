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


### To run `.sh` script:

Add any of the python commands you want to run, then make sure to run in the terminal (with the correct .venv environment activated:)
```
chmod +x run_scripts.sh
```

and run the script:
```
./run_scripts.sh
```

### Running `tmux`: (tips from ChatGPT)

To keep a terminal session running even if you disconnect or close the terminal, you can use `tmux`. It allows you to start a session that will continue running in the background even after you disconnect from the terminal. Here's how you can use `tmux`:

#### 1. **Start a new `tmux` session**
In your terminal, run:
```bash
tmux new-session -s mysession
```
This starts a new `tmux` session named `mysession`. You can replace `mysession` with any name you like.

#### 2. **Run your commands or scripts**
Once inside the `tmux` session, you can run your commands or scripts as usual. For example:
```bash
python script1.py
```

#### 3. **Detach from the `tmux` session**
To detach from the `tmux` session (leave it running in the background):
- Press `Ctrl` + `b` and then release both keys.
- Press `d` to detach.

Your session will continue running in the background.

#### 4. **Reattach to the `tmux` session**
To reattach to the `tmux` session:
```bash
tmux attach-session -t mysession
```

#### 5. **List all `tmux` sessions**
If you have multiple `tmux` sessions running, you can list them with:
```bash
tmux list-sessions
```

#### 6. **Kill a `tmux` session**
If you're done and want to terminate the session, you can kill it with:
```bash
tmux kill-session -t mysession
```

#### Additional tips:
- You can create multiple windows inside a single `tmux` session by pressing `Ctrl` + `b`, then `c` to create a new window.
- You can switch between windows with `Ctrl` + `b`, then use the arrow keys or the window number.

Using `tmux` is great for running long processes or scripts that you don’t want to be interrupted when you disconnect from the terminal.
