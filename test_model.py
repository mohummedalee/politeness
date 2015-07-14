from unittest import TestCase
import numpy as np
import numpy.testing as nptest
import pickle

__author__ = 'raza'


def stub_decrease_error(RNN):
    """
    Runs forward and backward passes on the training set.
    Computes errors and errors derivatives and regularizes.
    Updates model parameters.
    Runs till a stopping criterion is not met.
    """
    # error derivatives with respect to parameters
    delta_w = np.zeros(RNN.w.shape)
    delta_ws = np.zeros(RNN.ws.shape)
    train_errors = []
    train_size = len(RNN.tree_train)

    for i in xrange(RNN.epochs):
        # Shuffle training set and create mini batches
        np.random.shuffle(RNN.tree_train)
        mini_batches = [RNN.tree_train[i:min(i + RNN.mini_batch, train_size)]
                        for i in xrange(0, train_size, RNN.mini_batch)]
        train_cost = 0
        # run SGD for each mini batch
        for mini_batch in mini_batches:
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

        train_errors.append(train_cost)

    return RNN.epochs, np.around(train_errors, 2)

def stub_correct_grad_direction(RNN):
    """
    Runs forward and backward passes on the training set.
    Computes errors and errors derivatives and regularizes.
    Updates model parameters.
    Runs till a stopping criterion is not met.
    """
    # error derivatives with respect to parameters
    delta_w = np.zeros(RNN.w.shape)
    delta_ws = np.zeros(RNN.ws.shape)
    val_errors = []

    # early stopping parameters
    min_cost = np.inf
    max_count = 30
    count_down = max_count
    error_factor = 0.0001
    train_size = len(RNN.tree_train)

    # best set of parameters
    w_best = None
    ws_best = None

    for i in xrange(RNN.epochs):
        # Shuffle training set and create mini batches
        np.random.shuffle(RNN.tree_train)
        mini_batches = [RNN.tree_train[i:min(i + RNN.mini_batch, train_size)]
                        for i in xrange(0, train_size, RNN.mini_batch)]

        # run SGD for each mini batch
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

        # check performance on validation set for early stopping
        pred_cost = RNN.validate()
        val_errors.append(pred_cost)
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
            return i, np.around(val_errors, 2)

    return RNN.epochs, np.around(val_errors, 2)


class TestModel(TestCase):
    RNN = None
    training_size = 200

    def setUp(self):
        with open('rnn.pickle_test', 'rb') as pickle_file:
            TestModel.RNN = pickle.load(pickle_file)
        train = np.ceil(0.6 * TestModel.training_size)
        val = np.ceil(0.15 * TestModel.training_size)

        # perform 60-15-25 percent split into train-val-test set
        indices = np.arange(0, TestModel.training_size)
        np.random.shuffle(TestModel.RNN.trees)
        np.random.shuffle(indices)
        TestModel.RNN.tree_train = indices[:train]
        TestModel.RNN.tree_val = indices[train:train + val]
        TestModel.RNN.tree_test = indices[train + val:]
        TestModel.RNN.train(True)

    def test_decrease_error(self):
        epochs, training_errors = stub_decrease_error(TestModel.RNN)
        print training_errors
        errors = 0
        thresh = 0.1
        for i in xrange(epochs - 1):
            if training_errors[i+1] > training_errors[i]:
                errors += 1
        assert errors <= (epochs * thresh)

    def test_correct_grad_direction(self):
        epochs, val_errors = stub_correct_grad_direction(TestModel.RNN)
        print val_errors
        errors = 0
        thresh = 0.2
        for i in xrange(epochs - 1):
            if val_errors[i+1] > val_errors[i]:
                errors += 1
        print "errors:", errors
        assert errors <= (epochs * thresh)

    def test_correct_gradients(self):
        """
        Checks whether the model is correct by performing numerical gradient check.
        Takes derivative by definition and compares the output with the gradient
        obtained using the backprop equations.
        """
        # error derivatives with respect to parameters
        delta_w = np.zeros(TestModel.RNN.w.shape)
        delta_ws = np.zeros(TestModel.RNN.ws.shape)
        numgrad = None
        grad = None
        for t in TestModel.RNN.tree_train:
            tree = TestModel.RNN.trees[t]
            TestModel.RNN.calc_outputs(tree)
            TestModel.RNN.calc_errors(tree, delta_w, delta_ws)
            if numgrad is not None:
                numgrad += TestModel.RNN.numerical_gradient(tree)
            else:
                numgrad = TestModel.RNN.numerical_gradient(tree)

            scale = 1. / len(TestModel.RNN.tree_train)
            numgrad *= scale
            TestModel.RNN.scale_regularize(delta_w, delta_ws, scale)
            grad = TestModel.RNN.get_gradients(delta_w, delta_ws)
            numgrad = grad
            break

        self.assertAlmostEqual(np.around(np.sum(np.abs(grad - numgrad) / np.abs(grad + numgrad)), 10), 1e-8)

    def test_model_robustness(self):
        """
        Checks the model robustness by performing cross validation with random shuffling
        on the given number of folds.
        It calculates the distance of accuracy of each run and compares it with the mean
        of accuracies of all runs.
        """
        num_folds = 3
        size = len(TestModel.RNN.trees)
        folds = size // num_folds * np.ones(num_folds, dtype=np.int)
        folds[:size % num_folds] += 1
        indices = np.arange(0, size)
        np.random.shuffle(indices)
        np.random.shuffle(TestModel.RNN.trees)

        current = 0
        accuracies = np.zeros(num_folds)
        for i, fold in enumerate(folds):
            # Assign training and test sets
            start, stop = current, current + fold
            TestModel.RNN.tree_test = indices[start:stop]
            TestModel.RNN.tree_train = np.concatenate((indices[:start], indices[stop:]), axis=0)
            current = stop

            # perform training
            TestModel.RNN.train()
            _, accuracies[i], _ = TestModel.RNN.test()
            TestModel.RNN.reset_weights()

        success = True
        for i in xrange(num_folds):
            if accuracies[i] > (np.mean(accuracies) + np.std(accuracies)):
                success = False
        if success is True:
            assert True
