import numpy as np
import re, csv
import matplotlib as plt
from sklearn.decomposition.pca import PCA
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
        open('/media/syedraza/PERSONAL/Seventh Semester/FYP/Stanford_politeness_corpus/wikipedia.annotated.csv')))

del data[0]
data = np.array(data)
requests = data[:, 2]
y = map(float, data[:, 13])
y = [1*(each >= 0) for each in y]


v = CountVectorizer(stop_words='english')
x = v.fit_transform(requests)
print "x.shape[1]:", x.shape[1]
print x.toarray()


with open("data.csv", "wb") as f:
    for r in requests:
        terms = pre_process(r)
        f.write(', '.join(terms) + "\n")

clf = LinearSVC(C=20.0)
clf.fit(x, y)
labels = clf.predict(x)


correct = sum([l == t for l, t in zip(labels, y)])
print "correct: ", correct
print "percent:", (correct * 1. / x.shape[0]) * 100
print "count: ", x.shape[0]


'''
# applying PCA for dimensionality reduction
pca = PCA(n_components=2)
r_pca = pca.fit_transform(x.toarray())


# plotting for visualization
target_names = ['Dimension1', 'Dimension2']
plt.scatter(x.toarray())
plt.show()
'''

