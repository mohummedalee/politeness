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


def pre_process(string):
    # 1 remove non-sensical characters including spaces
    string = re.sub("[^a-zA-Z0-9.,;:!-@'?\s]|'s|'d|'t|'ll|'m|'re|'ve|", "", string)

    # 2 tokenize
    tokens = re.split("[\s.,;:!-@?]+", string)

    # 3 lowercase. porter stemmer does this later too
    tokens = [token.lower() for token in tokens]

    # 4 remove stop words, but first load them
    with open("stop_words.txt", "r") as f:
        str_stop = f.read()
    stop_words = re.split("[\s\n]+", str_stop)

    _terms = [token for token in tokens if token not in stop_words]

    '''
    # 5 stem using porter stemmer
    stemmer = PorterStemmer()
    _terms = [stemmer.stem(term) for term in _terms]
    '''
    return _terms


data = list(
    csv.reader(
        open('/home/mehvish/FYP/PycharmProjects/Classifier/politeness-master/wikipedia.annotated.csv')))

del data[0] #why?
data = np.array(data)
requests = data[:, 2]
y = map(float, data[:, 13])
y = [1*(each >= 0) for each in y]


v = CountVectorizer(stop_words='english')
x = v.fit_transform(requests)



print "x.shape[1]:", x.shape[1]

with open("data.csv", "wb") as f:
    for r in requests:
        terms = pre_process(r)
        f.write(', '.join(terms) + "\n")


# using linear SVC
# clf = LinearSVC(C=20.0)
# clf.fit(x, y)
# labels = clf.predict(x)

# using decision trees

#clf = GradientBoostingClassifier(verbose=1)
clf = DecisionTreeClassifier()

cv= ShuffleSplit(len(x.toarray()),n_iter=4,test_size=0.1)
scores = cross_validation.cross_val_score(clf,x.toarray(),y,cv= cv,n_jobs = -1)

print scores
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
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

