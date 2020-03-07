Kente Cloth Authentication
==============================

Culturally situated tools project for kente cloth authentication

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.testrun.org


--------

Git large file storage note
------------
```
brew install git-lfs
#  this repository deals with large files
```


Install and Build
------------
```
#  for Mac OS
brew install python3
pip3 install mkvirtualenv
git clone https://github.com/robinsonkwame/kente-cloth-authentication
cd ./kente-cloth-authentication
virtualenv kente
source kente/bin/activate
pip install -r requirements.txt
```

Run Project
------------
```
# Here we show how to generate a numerically balanced dataset,
# note that the fake kente distribution is restricted since samples are drawn
# from across 3 or 4 instanes, not the 10+ authentic kente instances.
#
# These values assume 3 fake kente source images and 14 authentic kente images
#
# So for 14*x = 3*y we have y = x 14/3 ~ 5x
# Let's take on the order of 1000 authentic kente, then we have ~355 instances
# from each fake kente, so lets fudge those numbers and have
# 
# 14*71=994 ~= 3*330=990 which is basically balanced, then
# Note: The width and height are set to the input size of ImageNet CNNs

# First we generate interim images ...
python src/data/make_dataset.py  makeinterim -i ./data/raw/ --seed 0 --width 224 --height 224 --target_width 224 --target_height 224 --xrotation 40 --yrotation 40 --zrotation 10 --number_per_real 71 --number_per_fake 330

# We then manually divide that up into train, validation and evaluation sets.

# ... then we erase any data interim with 
python src/data/make_dataset.py  cleanup



#  OLD ignore
# ... following the OC_NN MINST experiment, we generate
# about 6,000 normal cases and 60 (1%) anomalous cases
python src/data/make_dataset.py  --seed 0 --width 200 --height 200 --target_width 32 --target_height 32 --xrotation 40 --yrotation 40 --zrotation 10 --number_per_real 500 --number_per_fake 22

#  python ./src/models/train_model.py to train a model against that data, writes model to ./models too
#  python ./src/models/predict_model.py pulls from ./model and predicts
```


<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
