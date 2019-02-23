import os, math, random, pickle, urllib.parse

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

'''
Script for training the selected classifier (Logistic Regression)
Sentiment analysis in query_string_analyzer.py
'''

N = 3       # Ngram value
RANDOM_STATE = 123


def save_pickle(variable, filename):
    save_file = open(filename, 'wb')
    pickle.dump(variable, save_file)
    save_file.close()


def load_file(filename):
    directory = str(os.getcwd())
    filepath = os.path.join(directory, filename)
    result = []
    with open(filepath, 'r') as f:
        for line in f:
            data = str(urllib.parse.unquote(line))  # converting encoded url to simple string
            result.append(data)
    return list(set(result))    # delete duplicate datas


def n_gram_tokenizer(query_string):
    n_grams = []
    if len(query_string) <= N/2:      # too short, it's a token itself
        n_grams.append(query_string)
        return n_grams
    for i in range(0, len(query_string) - N):
        n_grams.append(query_string[i:i + N])
    return n_grams


def confidence_interval(model, x_validation, y_validation):
    # Const values: 1.64 (90%), 1.96 (95%), 2.33 (98%), 2.58 (99%)
    const = 2.58 #want 95% confidence interval
    incorrect_prediction = 0 #todo esce sempre 0, correggere
    y_predicted = model.predict(x_validation)
    for y_pred, y in zip(y_predicted, y_validation):
        if y_pred != y:
            incorrect_prediction += 1
    error = incorrect_prediction/len(y_validation)
    n = len(y_validation)
    value = const * math.sqrt((error * (1 - error)) / n)
    lower_bound = error - value
    upper_bound = error + value
    if lower_bound < 0:
        lower_bound = 0
    if upper_bound > 1:
        upper_bound = 1

    return lower_bound, upper_bound


def validate_model(model, x_validation, y_validation, model_name):
    print(f'\n\n----------------  {model_name}  ----------------')
    # Metrics evaluation
    print(f'Score: \t\t\t\t{model.score(x_validation, y_validation) * 100:10.2f}')
    print(f'Accuracy: \t\t\t{model.score(x_validation, y_validation)*100:10.2f}')
    lower_bound, upper_bound = confidence_interval(model, x_validation, y_validation)
    print(f'Confidence Interval: \t[{lower_bound:.2f}, {upper_bound:.2f}]')

def file_len(filename):
    count = 0
    with open(filename, "r") as file:
        for count, line in enumerate(file):
            pass
    return count + 1

def get_dork_payload_data_size(attack_size, attack):
    if attack == "X_PATH":
        dork_attack = "SQLi"
    else:
        dork_attack = attack

    dork_size = file_len(f"data/dorks/{dork_attack}.txt")
    payload_size = file_len(f"data/payloads/{attack}.txt")

    if dork_size*payload_size < attack_size:
        print(f"Not enough {attack} data! NEEDED: {attack_size}\nADDED ALL POSSIBLE!")
        return dork_size, payload_size

    needed_dorks = math.ceil(math.sqrt((3 / 7) * attack_size))
    needed_payloads = math.ceil(math.sqrt((7 / 3) * attack_size))

    if needed_dorks > dork_size or needed_payloads > payload_size:
        #dont have enough dorks or payloads, so get reault by iterative method
        min_val = min(dork_size, payload_size)
        if min_val == dork_size:
            #low dorks
            needed_dorks = dork_size
            needed_payloads = math.ceil(attack_size/needed_dorks)
        else:
            #low payloads
            needed_payloads = payload_size
            needed_dorks = math.ceil(attack_size/needed_payloads)

    return needed_dorks, needed_payloads

def add_vulnerability(size, attack):
    result = []
    if attack == "X_PATH":
        dork_attack = "SQLi"
    else:
        dork_attack = attack
    with open(f"data/dorks/{dork_attack}.txt", "r") as dorks, open(f"data/payloads/{attack}.txt", "r") as payloads:
        dorks = dorks.readlines()
        payloads = payloads.readlines()
    if len(dorks)*len(payloads) < math.ceil(0.1*size):
        #not enough data, create all we can
        print("NOT ENOUGH DATA")
        for dork in dorks:
            for payload in payloads:
                data = dork[:-1] + payload
                result.append(data)
    else:
        needed_dorks, needed_payloads = get_dork_payload_data_size(math.ceil(0.1*size), attack)
        dorks = random.sample(dorks, needed_dorks)
        payloads = random.sample(payloads, needed_payloads) #todo vedere cosa fa sample
        for dork in dorks:
            for payload in payloads:
                data = dork[:-1] + payload
                result.append(data)

    return result


def add_vulnerabilities(X_test, attack_list):
    new_vuln = []
    for attack in attack_list:
        new_vuln = new_vuln + add_vulnerability(len(X_test), attack)

    # todo vedere di cosa fa il replace:
    X_test = new_vuln + X_test[len(new_vuln):]
    random.shuffle(X_test)
    return X_test

def update_vulnerabilities(X_test, y_test, attack_list):
    new_X_test = add_vulnerabilities(X_test, attack_list)
    new_y_test = []
    for old_x, new_x, old_y in zip(X_test, new_X_test, y_test):
        if old_x == new_x:
            new_y_test.append(old_y)
        else:
            new_y_test.append(1)    #replaced with bad
    return new_X_test, new_y_test


# Creating inputs (X)
#good_queries = load_file('dataset/myDataset/good.txt')
#bad_queries = load_file('dataset/myDataset/bad.txt')


good_queries = load_file('resultTMP/good.txt')
bad_queries = load_file('resultTMP/bad.txt')

good_len = len(good_queries)
bad_len = len(bad_queries)

min_val = min(good_len, bad_len)
if min_val == good_len:
    bad_queries = bad_queries[:good_len]    #less good queries, so balance dataset
else:
    good_queries = good_queries[:bad_len]

all_queries = good_queries + bad_queries   # list of all queries, first the good one

print(f"GOOD QUERIES: {len(good_queries)}")
print(f"BAD QUERIES: {len(bad_queries)}")

# Create supervised output (y), 1 for bad queries, 0 for good queries
good_query_y = [0 for i in range(0, len(good_queries))]
bad_query_y = [1 for i in range(0, len(bad_queries))]
y = good_query_y + bad_query_y      # outputs vector, first for good queries

# Split dataset: train, validation and test (80,10,10)
X_train, X_test, y_train, y_test = train_test_split(all_queries, y, test_size=0.2, random_state=RANDOM_STATE)
X_validation, X_test, y_validation, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=RANDOM_STATE)



#todo try remove vuln in training (probability = 0.0)
#X_test = add_vulnerabilities(X_test, ["XSS"])
#X_test  = X_train[:len(X_test)]

#X_test, y_test = update_vulnerabilities(X_test, y_test, ["SSI"])
#i=0
#for x, y in zip(X_test, y_test):
    #print(f'[{i}] OUT:{y}\t{x}')
#    i+=1

vectorizer = TfidfVectorizer(tokenizer=n_gram_tokenizer)    # term frequency-inverse document frequency
vectorizer_fitted = vectorizer.fit(X_train)
save_pickle(vectorizer_fitted, "classifiers/TFIDF_Vectorizer")


X_train = vectorizer.transform(X_train)   # convert inputs to vectors
X_validation = vectorizer.transform(X_validation)
X_test = vectorizer.transform(X_test)



# -----------------------------MODEL SELECTION-----------------------------
# LogisticRegression
LogisticRegression_Classifier = LogisticRegression(solver='liblinear', penalty='l1', random_state=RANDOM_STATE, max_iter=200) # put solver to silence the warning
LogisticRegression_Classifier.fit(X_train, y_train)   # train the model
save_pickle(LogisticRegression_Classifier, 'pickle_tmp/LogisticRegression_Classifier.pickle')

# Metrics of all models (use Validation dataset):
validate_model(LogisticRegression_Classifier, X_validation, y_validation, 'Logistic Regression')

# Selected model: Logistic Regression
# Metrics of the selected model (use Test dataset):
validate_model(LogisticRegression_Classifier, X_test, y_test, 'BEST CLASSIFIER - LOGISTIC REGRESSION')
