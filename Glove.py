import numpy as np
import pickle

if __name__ == '__main__':
    vector_file = open('glove-50d.txt')

    word_vecs = {}
    for vector in vector_file:
        vec = vector.split()
        word = vec[0]
        word_vecs[word] = np.array(vec[0:])

    output = open('output.txt', 'w')
    pickle.dump(word_vecs, output, pickle.HIGHEST_PROTOCOL)
