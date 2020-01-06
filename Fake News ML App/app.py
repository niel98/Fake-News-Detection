from flask import Flask, render_template, url_for, request
import pandas as pd
import re
import string
import pickle

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods = ['POST'])
def predict():

#importing our already trained model using pickle
#    with open ('log_model', 'rb') as f:
#        lr_model = pickle.load(f)

    #importing the dataset as a corpus using the pandas library
    data = pd.read_csv('data.csv')
    data.head()

    #inspecting the data to see what it looks like
    data['Body'][0]

    data['Body'][:7]

    #Looking at the data, there are some missing columns, so let's take care of that
    data.fillna('Article unavailable')

    #data cleaning with text preprocessing techniques
    #data cleaning first round
    #using regular expressions and string to clean

    #function for first round of data cleaning
    def clean_text_round1(text):
        text = str(text).lower() #making all text lowercase
        text = re.sub('\[.*?\]', '', text) #removing full stops and question marks
        text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
        text = re.sub('\w*\d\w*', '', text) #removing digits
        return text

    round1 = lambda x: clean_text_round1(x)

    #Let's take a look at the updated text
    data_clean = pd.DataFrame(data.Body.apply(round1))
    data_clean

    #let's apply a second round of cleaning because some nonsensical text was ignored in the first clean
    def clean_text_round2(text):
        text = re.sub('[''""...]', '', text)
        text = re.sub('\n', '', text)
        return text

    round2 = lambda x: clean_text_round2(x)

    #let's take a look at the updated text again
    data_clean = pd.DataFrame(data_clean.Body.apply(round2))
    data_clean['Body'][0]

    #Concatenating our cleaned data to our corpus
    data['clean_body'] = data_clean
    data['clean_body'][0]

    #Extract features and target variables
    import numpy as np

    X = np.array(data['clean_body'], data['URLs'])     #feature variables
    y = np.array(data['Label'])                          #target variable

    y = list(map(int, y))

    #Split the data into folds
    from sklearn.model_selection import StratifiedKFold

    kf = StratifiedKFold(n_splits = 2)
    kf.get_n_splits(X)

    #Split the data into train and test
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33)

    #create a document-term matrix for the train and test data using tfidf vectorizer
    from sklearn.feature_extraction.text import TfidfVectorizer

    tf = TfidfVectorizer(stop_words = 'english', max_df = 0.7)        #removing all Englilsh stop words
    tf_train = tf.fit_transform(X_train)
    tf_test = tf.transform(X_test)

    #get feature names
    #tf.get_feature_names()

    #Now we feed our data into our classifiers to develop our model.
    #First we try the Naive Bayes classifier
    from sklearn.naive_bayes import MultinomialNB

    nb = MultinomialNB()

    #training the model
    nb.fit(tf_train, y_train)

    #predicting
    nb_pred = nb.predict(tf_test)
    nb_pred[0:10]

    #Evaluating the accuracy of the model
    nb_score = nb.score(tf_test, y_test)
    print('accuracy: %0.3f' % nb_score)

    #Next let's build another model using logistic Regression
    from sklearn.linear_model import LogisticRegression

    lr = LogisticRegression()

    #training the model
    lr.fit(tf_train, y_train)

    #predicting
    lr_pred = lr.predict(tf_test)
    lr_pred[0:10]

    #Evaluating the accuracy of the logistic regression model
    lr_score = lr.score(tf_test, y_test)
    print('accuracy: %0.3f' % lr_score)

    #Using the random forest classifier
    from sklearn.ensemble import RandomForestClassifier

    rf = RandomForestClassifier()

    #training the model
    rf.fit(tf_train, y_train)

    #predicting
    rf_pred = rf.predict(tf_test)
    rf_pred[0:10]

    #Evaluating the accuracy of the model
    rf_score = rf.score(tf_test, y_test)
    print('accuracy: %0.3f' % rf_score)

    #let's pickle the data_dtm for future use
    #data_dtm.to_pickle("dtm.pkl")

    #Since the logistic regression model is the most accurate classifier, lets save it and test it with
    #other news articles

    #saving the model

    with open ('log_model', 'wb') as f:
        pickle.dump(lr, f)

    #importing the model and testing

    with open ('log_model', 'rb') as f:
        lr_model = pickle.load(f)



    if request.method == 'POST':
        article = request.form['article']
        input_article = [article]
        vect = tf.transform(input_article).toarray()
        lr_predict = lr_model.predict(vect)

    return render_template('result.html', prediction = lr_predict)

if __name__ == '__main__':
    app.run(debug = True)
