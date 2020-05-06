# import libraries
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import nltk
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.multioutput import MultiOutputClassifier
import pickle
nltk.download('punkt')
nltk.download('wordnet')
import sys


def load_data(database_filepath):
    # load data from database
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql('SELECT * FROM disaster_response', engine)
    
    X = df.filter(items=['id', 'message', 'original', 'genre'])
    y = df.drop(['id', 'message', 'original', 'genre'], axis=1)
    
    #Mapping the '2' values in 'related' to '1' - because I consider them as a response (that is, '1')
    y['related']=y['related'].map(lambda x: 1 if x == 2 else x)
    
    return X, y

def tokenize(text):
    tokens = nltk.word_tokenize(text)
    lemmatizer = nltk.WordNetLemmatizer()
    return [lemmatizer.lemmatize(w).lower().strip() for w in tokens]


def build_model(X_train, X_test, y_train, y_test):
    pipeline = Pipeline([('cvect', CountVectorizer(tokenizer = tokenize)),
                     ('tfidf', TfidfTransformer()),
                     ('clf', RandomForestClassifier())
                     ])
    parameters = {'clf__max_depth': [None, 5, 10],
              'clf__min_samples_leaf': [1, 2],
              'clf__min_samples_split': [2, 5],
              'clf__n_estimators': [30, 40, 50]}

    cv = GridSearchCV(pipeline, param_grid=parameters, scoring='f1_micro', verbose=1, n_jobs=-1)
    cv.fit(X_train['message'], y_train)
    
    li = []
    for  i in cv.best_params_.values():
        li.append(i)
        
    pipeline = Pipeline([('cvect', CountVectorizer(tokenizer = tokenize)),
                     ('tfidf', TfidfTransformer()),
                     ('clf', RandomForestClassifier(max_depth = li[0], min_samples_leaf = li[1], min_samples_split = li[2], n_estimators = li[3]))
                     ])
    #X_train, X_test, y_train, y_test = train_test_split(X, y)
    #pipeline.fit(X_train['message'], y_train)
    return pipeline


def evaluate_model(pipeline, X_train, X_test, y_train, y_test, y):
    y_pred_test = pipeline.predict(X_test['message'])
    y_pred_train = pipeline.predict(X_train['message'])
    print(classification_report(y_test.values, y_pred_test, target_names=y.columns.values))
    print('...................................................................')
    print('\n',classification_report(y_train.values, y_pred_train, target_names=y.columns.values))


def save_model(model, model_filepath):
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)
    

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model(X_train, X_test, y_train, y_test)
        
        print('Training model...')
        model.fit(X_train['message'], y_train)
        
        print('Evaluating model...')
        evaluate_model(model,X_train, X_test, y_train, y_test, Y)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()