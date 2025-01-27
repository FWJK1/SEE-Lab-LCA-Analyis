# SEE-Lab-LCA-Analyis
Repository for brightway-based characterization and analysis of ecoinvent database

# Setup
## Environment
* First, make sure you have conda installed and running. 
* Next, enter `conda env create -f .\Environments\SEE_BW_environment.yml -n SEE_BW` in your terminal to make a a `SEE_BW` virtual environment with all the depencies.
* If you don't already have jupyter installed in your global environment, do that. Once you have it installed, use `python -m ipykernel install --user --name=SEE_BW --display-name "Python (SEE_BW)"` to set up the virtual environment you built from the yml so you can use the environment in your files.
* For .py scripts, first enter `conda activate SEE_BW` to make sure you're in the right environment.
* For .ipynb notebooks, make sure that you select the `SEE_BW` ipykernel to run the code successfully.
* **Note**: There might be new depencies since you last created the environment. If you have trouble, use `conda env remove -n SEE_BW` to first remove your old environment and then complete the steps above to make sure everything is in place.

## Security
* This project requires that you have an en ecoinvent license including the ability to download datasets. If you don't have this, sorry.
* In your local copy of this project, create a folder `/secrets`. Within this folder, create a file `passwords.json`. Fill out the file in this manner: 
    ```
    {
        "ecoinvent_username" : "your_username",
        "ecoinvent_password" : "your_password"
    }
    ```
* The secrets folder is included in the `.gitignore` file, so the info stays on your local machine.

## Steps
* First, run the 01_setup.ipynb notebook to build the SEE_LAB Brightway Project and load it with the ecoinvent database. For some reason this only seems to work in a jupyter notebook and not in a .py script. 