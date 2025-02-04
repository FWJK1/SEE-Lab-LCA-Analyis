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
* First, run the `Code/01_setup.ipynb` notebook to build the SEE_LAB Brightway Project and load it with the 3.9.1 and 3.11 ecoinvent cutoff databases. For some reason this seems to work better in a .ipynb jupyter notebook then in a .py script. 
* Second, make sure all your data is up to date. While I will attempt to host publically available and small indexing datasets in the repository, it's often better to build stuff if you can. See [Data Sources](#data-sources). 
* Next, go through the notebooks in `Exploratory_notebooks`  in order. Use the `SEE_BW` ipykernel you created in the [Environment](#environment) steps. While no later code relies upon these notebooks, they are helpful for understanding the proect.

# Data Sources
* [ecoQuery ecoinvent Database](https://ecoquery.ecoinvent.org/) for primary database

## Economic Data
* [World Bank](https://data.worldbank.org/) for GDP Data

## Indexing Data
* [Cloford.com] (https://cloford.com/resources/codes/index.htm) for indexing Country codes to sub-continental regions
* [Country and Continent Codes (stevewithington GitHub) ](https://gist.github.com/stevewithington/20a69c0b6d2ff846ea5d35e5fc47f26c) for indexing Country codes to Continents

## Shapefiles
* [Natural Earth](https://www.naturalearthdata.com/) for global state and territory shapefiles

