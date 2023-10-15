# Variability Tree for Unsupervised Classification of Variable Stars
Alice, Annya, Lina, Sage 2023

We use the variability tree method from [Valenzuela et al. 2017](https://academic.oup.com/mnras/article/474/3/3259/4622975?login=false) to try to classify variable stars from the ASAS-SN catalog. 

## Conventions
* snake_casing
* Epytext docstrings
## Data
* Raw data is individual .dat files for each object
* .dat files are converted to dataframes and pickled
* Pickled dataframes are used to construct Lightcurve objects