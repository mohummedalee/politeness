from unittest import TestCase
from Model import Model
import numpy as np
import pickle

__author__ = 'Myrrahh'

np.random.seed(43)


class TestModel(TestCase):
    def helper_ini(self, case):

        training_size = 0

        if case == 0:
            training_size = 10
        elif case == 1:
            training_size = 5
        elif case == 2:
            training_size = 0

        train = np.ceil(0.6 * training_size)
        val = np.ceil(0.15 * training_size)

        with open('rnn.pickle', 'rb') as pickle_file:
            RNN = pickle.load(pickle_file)

        indices = np.arange(0, training_size)
        np.random.shuffle(RNN.trees)
        np.random.shuffle(indices)
        RNN.tree_train = indices[:train]
        RNN.tree_val = indices[train:train + val]
        RNN.tree_test = indices[train + val:]

        RNN.train(True)
        return RNN

    def test_validate(self):
        """
        For checking the predictions of the validation function
        """
        for x in range(0, 3):
            RNN = self.helper_ini(x)
            pred = RNN.validate()
            val_cost = 0
            for t in RNN.tree_val:
                tree = RNN.trees[t]
                RNN.calc_outputs(tree)
                val_cost += RNN.get_cost(tree)

            assert np.allclose(pred, val_cost)

    def test_test(self):
        """
        for checking the accuracy of the predictions made on the testing set
        """
        for x in range(0, 2):
            RNN = self.helper_ini(0)
            a, b, c = RNN.test()

            correct = 0
            for t in RNN.tree_test:
                tree = RNN.trees[t]
                RNN.calc_outputs(tree)
                tree.pred_label = np.argmax(tree.predictions)
                true_label = np.where(tree.target == 1)[0]
                if true_label == tree.pred_label:
                    correct += 1

            assert np.allclose(b, 1. * correct / len(RNN.tree_test))

    def test_test1(self):
        """
        expected to fail
        """
        RNN = self.helper_ini(2)
        a, b, c = RNN.test()

        correct = 0
        for t in RNN.tree_test:
            tree = RNN.trees[t]
            RNN.calc_outputs(tree)
            tree.pred_label = np.argmax(tree.predictions)
            true_label = np.where(tree.target == 1)[0]
            if true_label == tree.pred_label:
                correct += 1

        assert np.allclose(b, 1. * correct / len(RNN.tree_test))