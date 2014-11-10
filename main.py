import numpy as np
import re, csv
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import cross_validation
from sklearn.cross_validation import ShuffleSplit

from sklearn.decomposition.pca import PCA
import matplotlib as plt
from nltk.stem.porter import PorterStemmer


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

if __name__ == '__main__':

    with open("stop_words.txt", "r") as f:
        str_stop = f.read()

    stop_words = re.split("[\s\n]+", str_stop)

    ###Getting wiki data
    wiki_data = list(csv.reader(open('wikipedia.annotated.csv')))

    del wiki_data[0] #remove field names

    wiki_data = np.array(wiki_data)

    wiki_requests = wiki_data[:, 2]
    wiki_requests = [pre_process(one) for one in wiki_requests]

    wiki_scores = map(float, wiki_data[:, 13]) #y
    wiki_scores = [1*(each >= 0) for each in wiki_scores]


    ###Getting SE data
    SE_data = list(csv.reader(open('stack-exchange.annotated.csv')))

    del SE_data[0] #remove field names

    SE_data = np.array(SE_data)

    SE_requests = SE_data[:, 2]
    SE_requests = [pre_process(one) for one in SE_requests]

    SE_scores = map(float, SE_data[:, 13]) #y
    SE_scores = [1*(each >= 0) for each in SE_scores]

    ##############################################################
    v = CountVectorizer(stop_words='english')
    v = v.fit(SE_requests)
    wiki_x = v.transform(wiki_requests)
    SE_x = v.transform(SE_requests) 

    print "wiki_x.shape[1]:", wiki_x.shape[1]
    print "SE_x.shape[1]:", SE_x.shape[1]

    clf = LinearSVC(C=20.0)
    clf.fit(wiki_x, wiki_scores)

    scores = clf.predict(SE_x)

    print "SVC c=20: ", scores
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

'''
    clf = LinearSVC(C=20.0)
    clf.fit(wiki_x, wiki_scores)
    labels = clf.scores(SE_x, SE_scores)

    print "SVC c=20 ", labels

    

    # using decision trees

    #clf = GradientBoostingClassifier(verbose=1)

    clf = DecisionTreeClassifier()

    cv= ShuffleSplit(len(wiki_x.toarray()),n_iter=2,test_size=0.1) #shuffle split commit

    scores = cross_validation.cross_val_score(clf, SE_x.toarray(), wiki_scores, cv= cv, n_jobs = -1)

    print scores
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
'''
    #print scores
    # clf = clf.fit(x.toarray(), y)
    #
    #
    #
    # labels = clf.predict(x.toarray())
    #
    # correct = sum([l == t for l, t in zip(labels, y)]) #matches the predicted labels with previous labels and sums
    # print "correct: ", correct
    # print "percent:", (correct * 1. / x.shape[0]) * 100
    # print "count: ", x.shape[0]


'''
    # applying PCA for dimensionality reduction
    pca = PCA(n_components=2)
    r_pca = pca.fit_transform(x.toarray())


    # plotting for visualization
    target_names = ['Dimension1', 'Dimension2']
    plt.scatter(x.toarray())
    plt.show()
'''

