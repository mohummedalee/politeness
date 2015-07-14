from Node import Node
from Tree import Tree
from utilities import *
import numpy as np
import pickle
from Model import Model
from unittest import TestCase

np.random.seed(43)


class RNNBranchCoverage:
    def Stub_validate(self, RNN):
        RNN.tree_val = np.array([2])
        val_cost = 0
        x = RNN.tree_val[0]
        for t in RNN.tree_val:
            tree = RNN.trees[t]
            RNN.calc_outputs(tree)
            val_cost += RNN.get_cost(tree)

        return val_cost

    def Stub_check_model_veracity(self, RNN):
        output = []
        RNN.epochs = 1
        RNN.tree_train = (2, 10)
        delta_w = np.zeros(RNN.w.shape)
        delta_ws = np.zeros(RNN.ws.shape)
        for i in xrange(RNN.epochs):
            numgrad = None
            for t in RNN.tree_train:
                tree = RNN.trees[t]
                RNN.calc_outputs(tree)
                RNN.calc_errors(tree, delta_w, delta_ws)
                if numgrad is not None:
                    numgrad += RNN.numerical_gradient(tree)
                else:
                    numgrad = RNN.numerical_gradient(tree)

            scale = 1. / len(RNN.tree_train)
            numgrad *= scale
            RNN.scale_regularize(delta_w, delta_ws, scale)
            grad = RNN.get_gradients(delta_w, delta_ws)
            output.append(np.around(np.sum(np.abs(grad - numgrad) / np.abs(grad + numgrad)), 10))

            RNN.update(delta_w, delta_ws)
            delta_w.fill(0)
            delta_ws.fill(0)

        return output

    def Stub_cross_validate(self, RNN, num_folds=1):
        size = len(RNN.trees)
        folds = size // num_folds * np.ones(num_folds, dtype=np.int)
        folds[:size % num_folds] += 1
        indices = np.arange(0, size)
        np.random.shuffle(indices)
        np.random.shuffle(RNN.trees)

        current = 0
        accuracies = np.zeros(num_folds)
        for i, fold in enumerate(folds):
            # Assign training and test sets
            start, stop = current, current + fold
            RNN.tree_test = indices[start:stop]
            RNN.tree_train = np.concatenate((indices[:start], indices[stop:]), axis=0)
            current = stop

            # perform training
            '''
            Irrelevant for branch coverage as take up alot of time
            RNN.train()
            _, accuracies[i], _ = RNN.test()
            RNN.reset_weights()
            '''
        return np.mean(accuracies)

    def Stub_test(self, RNN):
        RNN.tree_test = np.array([4, 10])
        test_cost = 0
        correct = 0
        incorrect = []
        for t in RNN.tree_test:
            tree = RNN.trees[t]
            RNN.calc_outputs(tree)
            test_cost += RNN.get_cost(tree)
            tree.pred_label = np.argmax(tree.predictions)
            true_label = np.where(tree.target == 1)[0]
            if true_label == tree.pred_label:
                correct += 1
            else:
                incorrect.append(tree.id)

        return 1. * correct / len(RNN.tree_test)

    # stub forward#1
    def Stub1_forward(self, node, RNN):
        """
          Checks true/false for first two and true for the third
        """

        if node.num_child == 0:
            return node.vec

        elif node.num_child == 1:
            return np.tanh(node.children[0].vec)

        elif node.num_child == 2:
            # setting number of children of left child to 1 to check true of the second condition
            node.children[0].num_child = 1
            left = RNN.forward(node.children[0])
            node.children[0].num_child = 0
            right = RNN.forward(node.children[1])
            children = concat_with_bias(left, right)

            node.vec = np.tanh(np.dot(RNN.w, children))

            return node.vec

        # stub forward#2

    def Stub2_forward(self, node, RNN):
        # For getting False from the third condition
        node.num_child = 3

        if node.num_child == 0:
            return node.vec

        elif node.num_child == 1:
            return np.tanh(node.children[0].vec)

        elif node.num_child == 2:
            left = RNN.forward(node.children[0])
            right = RNN.forward(node.children[1])
            children = concat_with_bias(left, right)

            node.vec = np.tanh(np.dot(RNN.w, children))

            return node.vec

    def Stub1_back_prop(self, RNN, node, delta_com, delta_w, delta_ws):
        """
        Checks true/false for first two and true for the third
        """

        if node.num_child == 0:
            # TODO: take word vector derivatives
            return
        elif node.num_child == 1:
            return
        elif node.num_child == 2:
            node.children[0].num_child = 1
            left_vector = node.children[0].vec
            node.children[1].num_child = 0
            right_vector = node.children[0].vec
            # [x3, p1]

            # concatenate with bias here
            children = concat_with_bias(left_vector, right_vector)

            # delta_w = delta_com * [x3, p1]
            delta_w += np.dot(delta_com, children.T)

            # W.T * delta_com * f'([x3, p1])
            delta_down = np.multiply(np.dot(RNN.w.T, delta_com), tanh_derivative(children))

            left_delta_down = delta_down[:RNN.dim]
            right_delta_down = delta_down[RNN.dim: 2 * RNN.dim]

            RNN.back_prop(node.children[0], left_delta_down, delta_w, delta_ws)
            RNN.back_prop(node.children[1], right_delta_down, delta_w, delta_ws)

    def Stub2_back_prop(self, RNN, node, delta_com, delta_w, delta_ws):
        """
        Checks false for third condition
        """
        node.num_child = 3

        if node.num_child == 0:
            # TODO: take word vector derivatives
            return
        elif node.num_child == 1:
            return
        elif node.num_child == 2:
            node.children[0].num_child = 1
            left_vector = node.children[0].vec
            node.children[1].num_child = 0
            right_vector = node.children[1].vec
            # [x3, p1]

            # concatenate with bias here
            children = concat_with_bias(left_vector, right_vector)

            # delta_w = delta_com * [x3, p1]
            delta_w += np.dot(delta_com, children.T)

            # W.T * delta_com * f'([x3, p1])
            delta_down = np.multiply(np.dot(RNN.w.T, delta_com), tanh_derivative(children))

            left_delta_down = delta_down[:RNN.dim]
            right_delta_down = delta_down[RNN.dim: 2 * RNN.dim]

            RNN.back_prop(node.children[0], left_delta_down, delta_w, delta_ws)
            RNN.back_prop(node.children[1], right_delta_down, delta_w, delta_ws)

    def Stub_numerical_gradient(self, RNN, tree):
        """
        Checks the only condition in the for loop
        """

        epsilon = 1e-5
        initial_params = RNN.get_params()
        RNN.set_params(initial_params)
        l = len(initial_params)

        # To run the loop only once
        l = 1

        vector = np.zeros(initial_params.shape)
        exp_grad = np.zeros(initial_params.shape)

        for i in range(l):
            vector[i] = epsilon

            RNN.set_params(initial_params + vector)
            RNN.calc_outputs(tree)
            c_plus = RNN.get_cost(tree)

            RNN.set_params(initial_params - vector)
            RNN.calc_outputs(tree)
            c_minus = RNN.get_cost(tree)

            exp_grad[i] = (c_plus - c_minus) / (2 * epsilon)

            vector[i] = 0

        RNN.set_params(initial_params)

        return exp_grad

    def Stub1_train(self, RNN, is_val=True):
        """

        """
        # error derivatives with respect to parameters
        delta_w = np.zeros(RNN.w.shape)
        delta_ws = np.zeros(RNN.ws.shape)
        train_cost = 0

        # early stopping parameters
        min_cost = np.inf
        max_count = 30
        count_down = max_count
        error_factor = 0.0001
        train_size = len(RNN.tree_train)

        # best set of parameters
        w_best = None
        ws_best = None

        RNN.epochs = 1

        for i in xrange(RNN.epochs):
            # Shuffle training set and create mini batches
            np.random.shuffle(RNN.tree_train)
            mini_batches = [RNN.tree_train[i:min(i + RNN.mini_batch, train_size)]
                            for i in xrange(0, train_size, RNN.mini_batch)]
            # run SGD for each mini batch
            mini_batches = [[123]]
            for mini_batch in mini_batches:
                train_cost = 0
                for t in mini_batch:
                    tree = RNN.trees[t]
                    # perform calculations
                    RNN.calc_outputs(tree)
                    RNN.calc_errors(tree, delta_w, delta_ws)
                    train_cost += RNN.get_cost(tree)

                # scale and regularize the parameters
                scale = 1. / len(mini_batch)
                RNN.scale_regularize(delta_w, delta_ws, scale)
                RNN.update(delta_w, delta_ws)

                # Reset the derivatives
                delta_w.fill(0)
                delta_ws.fill(0)

            if is_val:
                # check performance on validation set for early stopping
                pred_cost = RNN.validate()
                if pred_cost < (1 - error_factor) * min_cost:
                    min_cost = pred_cost
                    count_down = max_count
                    w_best = RNN.w.copy()
                    ws_best = RNN.ws.copy()
                else:
                    count_down -= 1

                # performance on validation set has not decreased significantly in the past
                if count_down == 0:
                    RNN.w = w_best
                    RNN.ws = ws_best
                    break

        return train_cost

    def Stub2_train(self, RNN, is_val=False):
        """
      For isVal=False
        """
        # error derivatives with respect to parameters
        delta_w = np.zeros(RNN.w.shape)
        delta_ws = np.zeros(RNN.ws.shape)
        train_cost = 0

        # early stopping parameters
        min_cost = np.inf
        max_count = 30
        count_down = max_count
        error_factor = 0.0001
        train_size = len(RNN.tree_train)

        # best set of parameters
        w_best = None
        ws_best = None

        RNN.epochs = 1

        for i in xrange(RNN.epochs):
            # Shuffle training set and create mini batches
            np.random.shuffle(RNN.tree_train)
            mini_batches = [RNN.tree_train[i:min(i + RNN.mini_batch, train_size)]
                            for i in xrange(0, train_size, RNN.mini_batch)]
            # run SGD for each mini batch
            mini_batches = [[123]]
            for mini_batch in mini_batches:
                train_cost = 0
                for t in mini_batch:
                    tree = RNN.trees[t]
                    # perform calculations
                    RNN.calc_outputs(tree)
                    RNN.calc_errors(tree, delta_w, delta_ws)
                    train_cost += RNN.get_cost(tree)

                # scale and regularize the parameters
                scale = 1. / len(mini_batch)
                RNN.scale_regularize(delta_w, delta_ws, scale)
                RNN.update(delta_w, delta_ws)

                # Reset the derivatives
                delta_w.fill(0)
                delta_ws.fill(0)

            if is_val:
                # check performance on validation set for early stopping
                pred_cost = RNN.validate()
                if pred_cost < (1 - error_factor) * min_cost:
                    min_cost = pred_cost
                    count_down = max_count
                    w_best = RNN.w.copy()
                    ws_best = RNN.ws.copy()
                else:
                    count_down -= 1

                # performance on validation set has not decreased significantly in the past
                if count_down == 0:
                    RNN.w = w_best
                    RNN.ws = ws_best
                    break

        return train_cost

    def Stub3_train(self, RNN, is_val=True):
        """
      For isVal=True,pred_cost condition False and count_down==0
        """
        # error derivatives with respect to parameters
        delta_w = np.zeros(RNN.w.shape)
        delta_ws = np.zeros(RNN.ws.shape)
        train_cost = 0

        # early stopping parameters
        min_cost = np.inf
        max_count = 30
        count_down = max_count
        error_factor = 0.0001
        train_size = len(RNN.tree_train)

        # best set of parameters
        w_best = None
        ws_best = None

        RNN.epochs = 1

        for i in xrange(RNN.epochs):
            # Shuffle training set and create mini batches
            np.random.shuffle(RNN.tree_train)
            mini_batches = [RNN.tree_train[i:min(i + RNN.mini_batch, train_size)]
                            for i in xrange(0, train_size, RNN.mini_batch)]
            # run SGD for each mini batch
            mini_batches = [[123]]
            for mini_batch in mini_batches:
                train_cost = 0
                for t in mini_batch:
                    tree = RNN.trees[t]
                    # perform calculations
                    RNN.calc_outputs(tree)
                    RNN.calc_errors(tree, delta_w, delta_ws)
                    train_cost += RNN.get_cost(tree)

                # scale and regularize the parameters
                scale = 1. / len(mini_batch)
                RNN.scale_regularize(delta_w, delta_ws, scale)
                RNN.update(delta_w, delta_ws)

                # Reset the derivatives
                delta_w.fill(0)
                delta_ws.fill(0)

            if is_val:
                # check performance on validation set for early stopping
                pred_cost = RNN.validate()
                pred_cost = (1 - error_factor) * min_cost - 1

                if pred_cost < (1 - error_factor) * min_cost:
                    min_cost = pred_cost
                    count_down = max_count
                    w_best = RNN.w.copy()
                    ws_best = RNN.ws.copy()
                else:
                    count_down -= 1

                # performance on validation set has not decreased significantly in the past
                count_down = 0
                if count_down == 0:
                    RNN.w = w_best
                    RNN.ws = ws_best
                    break

        return train_cost


class TestModel(TestCase):
    RNN = None
    R = None

    def setUp(self):
        TestModel.R = RNNBranchCoverage()
        training_size = 10

        train = np.ceil(0.6 * training_size)
        val = np.ceil(0.15 * training_size)

        with open('rnn.pickle_test', 'rb') as pickle_file:
            TestModel.RNN = pickle.load(pickle_file)

        indices = np.arange(0, training_size)
        np.random.shuffle(TestModel.RNN.trees)
        np.random.shuffle(indices)
        TestModel.RNN.tree_train = indices[:train]
        TestModel.RNN.tree_val = indices[train:train + val]
        TestModel.RNN.tree_test = indices[train + val:]
        TestModel.RNN.train(True)

    def test_Stub_validate(self):
        exp = 0.080
        actual = TestModel.R.Stub_validate(TestModel.RNN)
        self.assertAlmostEqual(exp, actual)

    def test_Stub_check_model_veracity(self):
        exp = 0.08999999999999999
        actual = TestModel.R.Stub_validate(TestModel.RNN)
        self.assertAlmostEqual(exp, actual)

    def test_Stub_test(self):
        exp = 1.
        actual = TestModel.R.Stub_test(TestModel.RNN)
        self.assertAlmostEqual(exp, actual)

    def test_Stub_cross_validate(self):
        exp = 0.
        actual = TestModel.R.Stub_cross_validate(TestModel.RNN)
        self.assertAlmostEqual(exp, actual)

    def test_Stub1_forward(self):
        exp = np.array([[-0.1006483 ],[-0.29548601],[ 0.0063629 ],[-0.17319958],[ 0.12327064],[ 0.02946587],[ 0.00644705],[-0.47383259],[-0.17706092],[ 0.14228462],[ 0.67255053],[-0.03523792],[ 0.2564717 ],[-0.18084416],[-0.3138467 ],[ 0.10138548],[-0.22682543],[ 0.11797612],[ 0.29268094],[ 0.17062235],[-0.19280561],[-0.31326626],[-0.05769646],[-0.31976983],[-0.1723143 ],[ 0.33203993],[-0.13410669],[ 0.04226759],[-0.27281455],[ 0.1945153 ],[-0.15207751],[ 0.32342922],[ 0.36387432],[ 0.10829055],[ 0.26577119],[ 0.08183803],[ 0.05923863],[-0.4977896 ],[-0.36396976],[-0.58843395],[ 0.51139022],[ 0.20288105],[-0.81887659],[ 0.21982588],[ 0.45401075],[-0.33887771],[-0.02298285],[-0.35496048],[-0.53704188], [0.27011948]])
        actual = TestModel.R.Stub1_forward(TestModel.RNN.trees[0].root, TestModel.RNN)
        assert np.allclose(exp, actual)
        # self.assertAlmostEquals(exp, actual)

    def test_Stub2_forward(self):
        exp = None
        actual = TestModel.R.Stub2_forward(TestModel.RNN.trees[0].root, TestModel.RNN)
        self.assertAlmostEqual(exp, actual)

    def test_Stub1_back_prop(self):
        exp = None
        actual = TestModel.R.Stub1_back_prop(TestModel.RNN, TestModel.RNN.trees[0].root, 0, 0, 0)
        self.assertAlmostEqual(exp, actual)

    def test_Stub2_back_prop(self):
        exp = None
        actual = TestModel.R.Stub2_back_prop(TestModel.RNN, TestModel.RNN.trees[0].root, 0, 0, 0)
        self.assertAlmostEqual(exp, actual)

    def test_Stub_numerical_gradient(self):
        exp = np.array([[ 0.],[ 0.],[ 0.],[ 0.],[ 0.],[ 0.],[ 0.],[ 0.],[ 0.],[ 0.],[ 0.],[ 0.],[ 0.],[ 0.],[ 0.],[ 0.],[ 0.],[ 0.]])
        actual = TestModel.R.Stub_numerical_gradient(TestModel.RNN, TestModel.RNN.trees[0])
        actual = actual[:18, :]
        assert np.allclose(exp, actual)

    def test_Stub1_train(self):
        exp = 0.089999999999999
        actual = TestModel.R.Stub1_train(TestModel.RNN, True)
        self.assertAlmostEqual(exp, actual)

    def test_Stub2_train(self):
        exp = 0.11
        actual = TestModel.R.Stub2_train(TestModel.RNN, False)
        self.assertAlmostEqual(exp, actual)

    def test_Stub3_train(self):
        exp = 0.13
        actual = TestModel.R.Stub3_train(TestModel.RNN, True)
        self.assertAlmostEqual(exp, actual)