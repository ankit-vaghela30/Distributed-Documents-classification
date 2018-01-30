import random

import numpy as np
import pyspark

from hydrus.math import RddTensor


class LogisticRegression:
    '''A model for multinomial logistic regression in Spark.
    '''

    def __init__(self, ctx):
        '''Initialize a LogisticRegression model from a SparkContext
        '''
        self.ctx = ctx
        self.weights = None
        self.feature = 0
        self.ids = None

    def fit(self, x, y, lr=0.01, batch_size=-1, max_iter=1, feature=0, warm_start=False):
        '''Train the model on some dataset and labels.

        Args:
            x: RDD ((id, feature), value)
                An RDD where `id` identifies each instance, `feature` names a
                feature of that instance, and `value` is the value of that
                feature for that instance. Missing values are considered 0.
            y: RDD (id, label)
                An RDD mapping instance IDs to true labels.
            lr: Positive float
                The learning rate.
            batch_size: int
                If greater than 0, sample the training set down to this size
                on each iteration. This will throw an exception if this is
                greater than the size of the training set.
            max_iter: Positive int
                The maximum number of epochs to train.
            feature: Positive int
                The index of the feature to learn from in the case that `x`
                has more than one feature.
            warm_start: bool
                If True, the weights are not (re)initialized.
        '''
        self.ids = y.keys().distinct().collect()
        self.ids = frozenset(self.ids)
        self.feature = feature

        x = RddTensor(x, 2, feature=feature)
        y = RddTensor.from_catigorical(y)

        if not warm_start:
            self.weights = self._weights(x, y)
            self.weights.persist(pyspark.StorageLevel.MEMORY_AND_DISK)

        for i in range(max_iter):
            if 0 < batch_size:
                x_, y_ = self._sample(x, y, batch_size)
            else:
                x_, y_ = x, y
            self.partial_fit(x_, y_, lr)

        return self

    def partial_fit(self, x, y, lr=0.01):
        '''Performs one iteration of gradient descent.

        Args:
            x: RDD ((id, feature), value)
                An RDD where `id` identifies each instance, `feature` names a
                feature of that instance, and `value` is the value of that
                feature for that instance. Missing values are considered 0.
            y: RDD (id, label)
                An RDD mapping instance IDs to true labels.
            lr: Positive float
                The learning rate.
        '''
        def debug(x):
            print('FITTING')
            return x

        old_weights = self.weights
        grad = self._gradient(x, y)
        self.weights -= grad * lr
        self.weights.persist(pyspark.StorageLevel.MEMORY_AND_DISK)
        old_weights.unpersist()
        return self

    def predict(self, x):
        '''Predict labels for some dataset.

        Args:
            x: RDD ((id, feature), value)
                An RDD where `id` identifies each instance, `feature` names a
                feature of that instance, and `value` is the value of that
                feature for that instance. Missing values are considered 0.

        Returns: RDD (id, label)
            An RDD mapping IDs to predicted labels.
        '''
        x = RddTensor(x, 2, feature=self.feature)
        h = (x @ self.weights).argmax().rdd   # ((doc_id,), label)
        h = h.map(lambda x: (x[0][0], x[1]))  # (doc_id, label)
        return h

    def score(self, x, y):
        '''Scores the predicted labels for x against the true labels y.

        This method currently only supports accuracy as a metric.

        Args:
            x: RDD ((id, feature), value)
                An RDD where `id` identifies each instance, `feature` names a
                feature of that instance, and `value` is the value of that
                feature for that instance. Missing values are considered 0.
            y: RDD (id, label)
                An RDD mapping instance IDs to true labels.

        Returns: float
            The accuracy of the prediction.
        '''
        def seq(score, labels):
            (id, (predicted, true)) = labels
            (correct, total) = score
            if predicted == true:
                return (correct + 1, total + 1)
            else:
                return (correct, total + 1)

        def comb(s, t):
            (correct_s, total_s) = s
            (correct_t, total_t) = t
            return (correct_s + correct_t, total_s + total_t)

        h = self.predict(x)
        pairs = h.join(y)
        (correct, total) = pairs.aggregate((0,0), seq, comb)
        return correct / total

    def _gradient(self, x, y):
        '''Computes the gradient of the cross-entropy loss w.r.t. the weights.

        Returns: RddTensor ((feature, label), grad)
            An RddTensor with the shape of the weights, where each element is
            the derivative of the loss w.r.t. the weight in the same position.
        '''
        pred = (x @ self.weights).softmax()
        delta = y - pred
        grad = -(x.t @ delta)
        return grad

    def _weights(self, x, y, n=1024):
        '''Create a weight matrix for mapping inputs like x to outputs like y.

        The values are drawn from a standard normal distribution.

        Args:
            x: RddTensor ((id, feature), value)
                An example of inputs to the regression. It must contain at
                least one instance of every feature.
            y: RddTensor ((id, label), 1)
                An example of the outputs of the regression, as an indicator
                matrix where the value 1 indicates the identified instance
                has the given label.
            n: Positive int
                The number of distinct initial values. They are generated as a
                float64 array. Every 512 values occupies 4KiB. In other words,
                use a multiple of 512 to keep the array page aligned on x86.

        Returns: RddTensor ((feature, label), weight)
            A matrix to transform feature columns into label columns.
        '''
        # To avoid enumerating all elements of the weight matrix and to avoid
        # using `np.random` in a Spark map function, we generate a fixed
        # amount of random numbers to use as initial weights.
        w = np.random.normal(size=n)
        w = self.ctx.broadcast(w)

        features = x.rdd.keys().values().distinct()
        labels = y.rdd.keys().values().distinct()
        weights = features.cartesian(labels)
        weights = weights.map(lambda x: (x, w.value[hash(x) % n]))
        weights = RddTensor(weights, 2)
        return weights

    def _sample(self, x, y, n):
        # Note we must use the standard library random instead of numpy because
        # self.ids has structure that numpy wants to treat as multidimensional.
        choices = random.sample(self.ids, n)
        choices = self.ctx.broadcast(choices)
        sample = lambda x: x[0][0] in choices.value
        x = x.rdd  # ((id, feature), x)
        y = y.rdd  # ((id, label), 1)
        x = x.filter(sample)
        y = y.filter(sample)
        x = RddTensor(x, 2)
        y = RddTensor(y, 2)
        return x, y
