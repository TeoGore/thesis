import pickle, urllib.parse
from sklearn.feature_extraction.text import TfidfVectorizer


'''
Sentiment analysis module for query string, (for training the model -->query_string_analyzer_PICKLE.py)
'''

N = 3       # Ngram value
RANDOM_STATE = 123


def load_pickle(filename):
    load_file = open(filename, 'rb')
    classifier = pickle.load(load_file)
    load_file.close()
    return classifier


def n_gram_tokenizer(query_string):
    n_grams = []
    if len(query_string) <= N/2:      # too short, it's a token itself
        n_grams.append(query_string)
        return n_grams
    for i in range(0, len(query_string) - N):
        n_grams.append(query_string[i:i + N])
    return n_grams


LogisticRegressionClassifier = load_pickle("classifiers/LogisticRegression_Classifier.pickle")
vectorizer = load_pickle("classifiers/TFIDF_Vectorizer")
# X = vectorizer.fit_transform(all_queries)   # convert inputs to vectors


def pre_process(query):
    data = str(urllib.parse.unquote(query))     # converting encoded url to simple string
    X = vectorizer.transform(data)      #todo correct errors
    return X


def sentiment(query):
    data = pre_process(query)
    return LogisticRegressionClassifier.predict(data)
#todo ritornare tupla (sentiment, accuracy)

#TESTING
print(sentiment("/google.com/"))
print(sentiment("/google.com/../../../***/"))
print(sentiment("/google.com/?cmd=cat /etc/passwd"))
