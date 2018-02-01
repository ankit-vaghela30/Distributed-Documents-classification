import numpy as np
import scipy as sp
import scipy.stats


class GaussianNaiveBayes:
    '''A model for Gaussian naive Bayes classification in Spark.
    '''

    def __init__(self, ctx):
        '''Initialize a GaussianNaiveBayes model from a SparkContext
        '''
        self.ctx = ctx

    def fit(self, x, y):
        '''Train the model on some dataset and labels.

        Args:
            x: RDD ((id, feature), value)
                An RDD where `id` identifies each instance, `feature` names a
                feature of that instance, and `value` is the value of that
                feature for that instance. Missing values are considered 0.
            y: RDD (id, label)
                An RDD mapping instance IDs to true labels.
        '''
        # Enumerate the labels, keep as an RDD
        vals = y.values()
        labels = vals.distinct()

        # Compute the label priors
        n = vals.count()
        priors = vals.countByValue()  # {label: count}
        priors = {k:v/n for k,v in priors.items()}  # {label: prior}

        # y is small-ish (number of documents).
        # Collect it to use in the RDD operations
        y = y.collectAsMap()  # {id: label}
        y = self.ctx.broadcast(y)

        # View the features both by doc id and by label.
        def doc_to_label(x):
            ((doc_id, feature), value) = x
            label = y.value[doc_id]
            return ((label, feature), value)
        by_label = x.map(doc_to_label)  # ((label, feature), value)

        # Compute the distribution of features.
        # We simultaneously reduce out the count, mean, and variance
        # with a tree aggregation. This is Chan's algorithm [1][2].
        # [1]: https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
        # [2]: http://i.stanford.edu/pub/cstr/reports/cs/tr/79/773/CS-TR-79-773.pdf
        def seq(a, b):
            b = (1, b, 0)
            return comb(a, b)
        def comb(a, b):
            (count_a, mean_a, var_a) = a
            (count_b, mean_b, var_b) = b
            count = count_a + count_b
            delta = mean_b - mean_a
            mean = mean_a + delta * count_b / count
            m2_a = var_a * (count_a - 1)
            m2_b = var_b * (count_b - 1)
            m2 = m2_a + m2_b + delta**2 * count_a * count_b / count
            var = m2 / (count - 1)
            return (count, mean, var)
        def var_to_stdev(x):
            (count, mean, var) = x
            stdev = np.sqrt(var)
            return (count, mean, stdev)
        stats = by_label.aggregateByKey((0,0,0), seq, comb)  # ((label, feature), (count, mean, var))
        stats = stats.mapValues(var_to_stdev)                # ((label, feature), (count, mean, stdev))

        # For naive bayes, we need the list of labels,
        # their priors, and the distributions of features.
        self.labels = labels
        self.priors = priors
        self.stats = stats
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
        # Cross and rekey by label
        def key_by_label(a):
            (label, ((id, feature), value)) = a
            return ((label, feature), (id, value))
        # x has initial shape ((id, feature), value)
        x = self.labels.cartesian(x)  # (label, ((id, feature), value))
        x = x.map(key_by_label)  # ((label, feature), (id, value))

        # Compute the conditionals, reduce to a ranking
        log_priors = {k:np.log(v) for k,v in self.priors.items()}
        log_priors = self.ctx.broadcast(log_priors)
        norm = sp.stats.norm
        def log_probs(a):
            ((label, feature), ((id, value), (count, mean, stdev))) = a
            if mean < value: value = mean - (value - mean)
            cd = norm.cdf(value, loc=mean, scale=stdev)
            prob = cd * 2
            log_prob = np.log(prob)
            return ((id, label), log_prob)
        def rank(a):
            ((id, label), log_prob) = a
            rank = log_prob + log_priors.value[label]
            return ((id, label), rank)
        x = x.join(self.stats)  # ((label, feature), ((id, value), (count, mean, stdev)))
        x = x.map(log_probs)  # ((id, label), log_prob)
        x = x.reduceByKey(lambda a, b: a + b)  # ((id, label), log_prob)
        x = x.map(rank, preservesPartitioning=True)  # ((id, label), rank)

        # Max out the best label
        def key_by_id(a):
            ((id, label), rank) = a
            return (id, (label, rank))
        def max_label(a, b):
            (label_a, rank_a) = a
            (label_b, rank_b) = b
            return b if rank_a < rank_b else a
        def flatten(a):
            (id, (label, rank)) = a
            return (id, label)
        x = x.map(key_by_id)  # (id, (label, rank))
        x = x.reduceByKey(max_label)  # (id, (label, rank))
        x = x.map(flatten)
        return x

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
