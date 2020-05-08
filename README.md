# Disaster Response Pipeline Project

###Contents:<br>
    Instructions<br>
    Installation<br>
    File Description<br>


### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### Installations
    Conda distribution distribution Python 3.x
    Microsoft Excel
    Flask
    
###File Description<br>
    messages_categories.csv: contains all messages data <br>
    disaster_categories.csv: contains categories data <br>
    DisasterResponse.db: It contains the cleaned and merged data of messages and categories <br>
    process_data.py: contains python codes to clean and merge the data. This data is stored in DisasterResponse.db <br>
    train_classifier.py: This will create and evaluate models required <br>
    run.py: Webapp's frontend codes 
    