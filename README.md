# Disaster-Response-Pipeline-Project
## Table of Contents
1. [ Introduction ](#intro)
2. [ Installation ](#installation)
3. [ Instructions ](#instructions)
4. [ Acknowledgements ](#acknowledgements)

<a name="intro"></a>
## 1. Introduction
The purpose of the project is to build a model for an API that classifies disaster messages. Using the web app we can input a new message and get classification results in several categories so to have an idea what kind of help is needed.


<a name="installation"></a>
## 2. Installation
Clone the GitHub repository and use Anaconda distribution of Python 3.9.7 <br>
```$ git clone https://github.com/FathimaSara/Disaster-Response-Pipeline-Project.git``` <br>
Libraries <br>
```$ pip install library_name```

Libraries Used <br>
NumPy, Pandas, Matplotlib, Json, Plotly, Nltk, Flask, Sklearn, Sqlalchemy, Sys, Re, Pickle

<a name="instructions"></a>
## 3. Instructions

Run the following commands in the project's root directory to set up your database and model.<br>

To run ETL pipeline that cleans data and stores in database:<br>
```$ python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db``` <br>

To run ML pipeline that trains classifier and saves model: <br>
```$ python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl``` <br>
 
Run the following command: model in the app's directory to run your web app: <br>
```$ python app/run.py``` <br>

 Go to http://0.0.0.0:3001


<a name="acknowledgements"></a>
## 4. Acknowledgements
Code templates and data were provided by Udacity from Figure Eight.I would like to express my gratitude towards Udacity and mentors for helping me in the project.



