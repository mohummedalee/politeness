import numpy as np
import re, csv
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import cross_validation, linear_model
from sklearn.cross_validation import ShuffleSplit
from sklearn.naive_bayes import MultinomialNB
from sklearn.decomposition.pca import PCA
import matplotlib as plt
from nltk.stem.porter import PorterStemmer
from sklearn.neighbors import KNeighborsClassifier

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC, SVC

global stop_words

def pre_process(string):
    # 1 remove non-sensical characters including spaces
    string = string.lower()
    string = re.sub("[^a-zA-Z0-9.,;:!-@'?\s]|'s|'d|'t|'ll|'m|'re|'ve|", "", string)
    # 2 tokenize
    tokens = re.split("[\s.,;:!-@?]+", string)
    # 3 remove stop words, but first load them
    _terms = [token for token in tokens if token not in stop_words]
    # 5 stem using porter stemmer
    stemmer = PorterStemmer()
    _terms = [stemmer.stem(term) for term in _terms]
    return " ".join(_terms)


# def print_scores(labels,wiki_scores,name):
#
#     correct = sum([l == t for l, t in zip(labels, wiki_scores)]) #matches the predicted labels with previous labels and sums
#     print "correct" + name+":" , correct
#     print "percent" + name+":", (correct * 1. / wiki_x.shape[0]) * 100


def InDomain(x,scores_,clf):

    cv= ShuffleSplit(len(x),n_iter=1,test_size=0.1)

    print "testing..."
    scores = cross_validation.cross_val_score(clf,x, scores_, cv= cv, n_jobs=-1, verbose=1)

    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

if __name__ == '__main__':

    with open("stop_words.txt", "r") as f:
        str_stop = f.read()

    stop_words = re.split("[\s\n]+", str_stop)

    ###Getting wiki data
    # wiki_data = list(csv.reader(open('wikipedia.annotated.csv')))
    #
    # del wiki_data[0] #remove field names
    #
    # wiki_data = np.array(wiki_data)
    #
    # wiki_requests = wiki_data[:, 2]
    # wiki_requests = [pre_process(one) for one in wiki_requests]
    #
    # wiki_scores = map(float, wiki_data[:, 13]) #y
    # wiki_scores = np.array([1*(each >= 0) for each in wiki_scores])


    ###Getting SE data
    SE_data = list(csv.reader(open('stack-exchange.annotated.csv')))

    del SE_data[0] #remove field names

    SE_data = np.array(SE_data)

    SE_requests = SE_data[:, 2]
    SE_requests = [pre_process(one) for one in SE_requests]

    SE_scores = map(float, SE_data[:, 13]) #y
    SE_scores = [1*(each >= 0) for each in SE_scores]
    SE_scores = np.array(SE_scores)
    ##############################################################
    v = CountVectorizer(stop_words='english')
    SE_x = v.fit_transform(SE_requests)

    #wiki_x = v.transform(wiki_requests)
    #wiki_x = np.array(wiki_x.toarray())
    SE_x = SE_x.toarray()

    #print "wiki_x.shape[1]:", wiki_x.shape[1]
    #print "SE_x.shape[1]:", SE_x.shape[1]

    # clf = SVC(kernel='rbf')
    # clf.fit(SE_x, SE_scores)
    #
    # scores = clf.predict(wiki_x)
    #
    # print "SVC RBF: ", scores
    # print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    #using gradient booster (without any cross validation)
    '''
    print "GradientBoostingClassifier..."
    GBClassifier = GradientBoostingClassifier()
    clf = GBClassifier.fit(SE_x,SE_scores)
    # labels = clf.predict(wiki_x.toarray())
    #
    # correct = sum([l == t for l, t in zip(labels, wiki_scores)]) #matches the predicted labels with previous labels and sums
    # print "correct: ", correct
    # print "percent:", (correct * 1. / wiki_x.shape[0]) * 100
    #print "count: ", x.shape[0]

# custom code for 90-10 split for cross validation
#     indices = np.arange(0, wiki_x.shape[0])
#     indices = np.random.shuffle(indices)
#     wiki_x = wiki_x[indices, :]
#     wiki_scores = wiki_scores[indices, :]
#
#     split = int(0.9 * wiki_x.shape[0])
#     train_wiki_x = wiki_x[:split]
#     test_wiki_x = wiki_x[split:]
#     train_wiki_scores = wiki_scores[:split]
#     test_wiki_scores = wiki_scores[split:]
#

    #using gradient booster (with in domain cross validation)
    '''
    
    

    # #using gradient booster (with cross domain cross validation)
    #
    # labels = clf.predict(SE_x)
    # correct = sum([l == t for l, t in zip(labels, wiki_scores)]) #matches the predicted labels with previous labels and sums
    # print "correct: ", correct
    # print "percent:", (correct * 1. / wiki_x.shape[0]) * 100
    #


    #using linear SVC (without any cross validation)
    '''
    print "LinearSVC..."
    clf = LinearSVC(C=20.0)
    clf.fit(SE_x, SE_scores)
    # labels = clf.predict(wiki_x)
    #
    # print_scores(labels,wiki_scores)


    #using linear SVC (with in domain cross validation)

    InDomain(SE_x,SE_scores,clf)
    # cv= ShuffleSplit(len(wiki_x),n_iter=3,test_size=0.1)
    # scores = cross_validation.cross_val_score(clf,wiki_x,wiki_scores,cv= cv,n_jobs = -1)
    # print scores
    # print("Accuracy (LinearSVC): %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    # #using linear SVC (with cross domain cross validation)



    #using Multinomial Naive Bayes (without any cross validation)

    print "MultinomialNB"
    clf =  MultinomialNB()
    clf.fit(SE_x,SE_scores)
    # labels = clf.predict(wiki_x)
    #
    # print_scores(labels,wiki_scores,"Naive Bayes")

    #using Multinomial Naive Bayes (in domain cross validation)

    InDomain(SE_x,SE_scores,clf)

    #using Multinomial Naive Bayes (cross domain cross validation)




    #using K Nearest Neighbour classifier (without cross validation)

    #3 neighbours

    print "3 KNeighborsClassifier..."

    clf = KNeighborsClassifier(n_neighbors=3)
    clf.fit(SE_x,SE_scores)
    # labels = clf.predict(wiki_x)
    #
    # print_scores(labels,wiki_scores,"3NN")

    #5 neighbours

    InDomain(SE_x,SE_scores,clf)

    print "5 KNeighborsClassifier..."

    clf = KNeighborsClassifier(n_neighbors=5)
    clf.fit(SE_x,SE_scores)
    # labels = clf5.predict(wiki_x)
    #
    # print_scores(labels,wiki_scores,"5NN")

    #using K Nearest Neighbour classifier (in domain cross validation)

    InDomain(SE_x,SE_scores,clf)

    #using K Nearest Neighbour classifier (cross domain cross validation)
    '''

    print "GradientBoostingClassifier..."
    clf = GradientBoostingClassifier()
    clf = clf.fit(SE_x,SE_scores)
    InDomain(SE_x,SE_scores,clf)

    #using Random Forest classifier (without cross validation)
    
    print "RandomForestClassifier..."
    clf = RandomForestClassifier(n_estimators=100)
    clf = clf.fit(SE_x,SE_scores)
    # labels = forest.predict(wiki_x)
    #
    # print_scores(labels,wiki_scores,"Random Forest")

    #using Random Forest classifier (without cross validation)

    InDomain(SE_x,SE_scores,clf)

    #using Random Forest classifier (without cross validation)
    print "Radial Basis Function..."
    clf = SVC(kernel='rbf')
    clf = clf.fit(SE_x,SE_scores)

    InDomain(SE_x,SE_scores,clf)

    #using Decision trees (without any cross validation)
    print "DecisionTreeClassifier..."
    clf = DecisionTreeClassifier()
    clf.fit(SE_x,SE_scores)
    InDomain(SE_x,SE_scores,clf)


    
    # labels = clf.predict(wiki_x)
    #
    # correct = sum([l == t for l, t in zip(labels, wiki_scores)]) #matches the predicted labels with previous labels and sums
    # print "correct (Decision trees): ", correct
    # print "percent (Decision trees):", (correct * 1. / wiki_x.shape[0]) * 100
    #
    # #using Decision trees (with in domain any cross validation)
    #
    # cv= ShuffleSplit(len(wiki_x),n_iter=3,test_size=0.1)
    # scores = cross_validation.cross_val_score(clf,wiki_x,wiki_scores,cv= cv,n_jobs = -1)
    #
    # print scores
    # print("Accuracy (Decision trees): %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    # #using Decision trees (with cross domain cross validation)
    #

