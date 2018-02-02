import argparse
import pyspark

import hydrus
from hydrus.preprocess import Loader, TfIdfTransformer


def get_context():
    '''Get the current SparkContext.
    '''
    global sc
    if 'sc' not in globals():
        sc = hydrus.interactive()
    return sc


def preprocess(args):
    '''Inspect the output of the data loader and TF-IDF transformer.
    '''
    counts, train_y = Loader(ctx).read(args.train_x, args.train_y)
    tfidfs = TfIdfTransformer(ctx).fit(counts).transform(counts)

    print('Data sample: ', counts.take(1)[0])
    if train_y is not None:
        print('Label sample:', train_y.take(1)[0])

    if args.all:
        print('All data:')
        def print_rdd(x):
            ((doc, word), (count, tfidf)) = x
            print('{doc:8}'.format(locals()), end='\t')
            print('{word:15}'.format(locals()), end='\t')
            print('{count:3}'.format(locals()), end='\t')
            print('{tfidf:3}'.format(locals()), end='\t')
            print()
        data = sounts.join(train_y)
        data.foreach(lambda x: print_rdd(x))


def softmax(args):
    '''Perform a logistic regression.
    '''
    train_x = args.train_x
    train_y = args.train_y
    test_x = args.test_x
    test_y = args.test_y
    lr = float(args.lr)
    batch_size = int(args.batch_size)
    iters = int(args.iters)

    ctx = get_context()
    ctx.setCheckpointDir('gs://hydrus/checkpoints')
    loader = hydrus.preprocess.Loader(ctx)
    tfidf = hydrus.preprocess.TfIdfTransformer(ctx)

    train_x, train_y = loader.read(train_x, train_y)
    if args.balance:
        train_x, train_y = hydrus.preprocess.sample_balanced(train_x, train_y)
    if args.tfidf:
        train_x = tfidf.fit(train_x).transform(train_x)

    test_x, test_y = loader.read(test_x, test_y)
    if args.tfidf:
        test_x = tfidf.transform(test_x)

    model = hydrus.logistic.LogisticRegression(ctx)
    model.fit(train_x, train_y, lr=lr, batch_size=batch_size, max_iter=iters)

    if test_y:
        score = model.score(test_x, test_y)
        print(score)
    else:
        pred = model.predict(test_x)
        hydrus.postprocess.print_labels(pred)


def gausian_naive_bayes(args):
    '''Perform a Gaussian naive Bayes classification.
    '''
    train_x = args.train_x
    train_y = args.train_y
    test_x = args.test_x
    test_y = args.test_y

    ctx = get_context()
    loader = hydrus.preprocess.Loader(ctx)
    tfidf = hydrus.preprocess.TfIdfTransformer(ctx)

    train_x, train_y = loader.read(train_x, train_y)
    if args.balance:
        train_x, train_y = hydrus.preprocess.sample_balanced(train_x, train_y)
    if args.tfidf:
        train_x = tfidf.fit(train_x).transform(train_x)

    test_x, test_y = loader.read(test_x, test_y)
    if args.tfidf:
        test_x = tfidf.transform(test_x)

    model = hydrus.naive_bayes.GaussianNaiveBayes(ctx)
    model.fit(train_x, train_y)

    if test_y:
        score = model.score(test_x, test_y)
        print(score)
    else:
        pred = model.predict(test_x)
        hydrus.postprocess.print_labels(pred)


def naive_bayes(args):
    '''Perform a regulr naive bayes classification
    '''
    train_x = args.train_x
    train_y = args.train_y
    test_x = args.test_x
    test_y = args.test_y

    ctx = get_context()
    loader = hydrus.preprocess.Loader(ctx)
    tfidf = hydrus.preprocess.TfIdfTransformer(ctx)

    train_x, train_y = loader.read(train_x, train_y)
    if args.balance:
        train_x, train_y = hydrus.preprocess.sample_balanced(train_x, train_y)
    if args.tfidf:
        train_x = tfidf.fit(train_x).transform(train_x)

    test_x, test_y = loader.read(test_x, test_y)
    if args.tfidf:
        test_x = tfidf.transform(test_x)

    model = hydrus.naive_bayes.NaiveBayes(ctx)
    model.fit(train_x, train_y)

    if test_y:
        score = model.score(test_x, test_y)
        print(score)
    else:
        pred = model.predict(test_x)
        print('pred is: ',pred.collect())
        hydrus.postprocess.print_labels(pred)

def info(args):
    '''Print system info.
    '''
    import sys
    print('Python version:')
    print(sys.version)


def main():
    parser = argparse.ArgumentParser(description='Execute hydrus commands')
    subcommands = parser.add_subparsers()

    # hydrus info
    cmd = subcommands.add_parser('info', description='print system info')
    cmd.set_defaults(func=info)

    # hydrus softmax [<args>...] <train_x> <train_y> <test_x> [<test_y>]
    cmd = subcommands.add_parser('softmax', description='multinomial logistic regression')
    cmd.add_argument('train_x', help='path to the training set')
    cmd.add_argument('train_y', help='path to the training labels')
    cmd.add_argument('test_x', help='path to the test set')
    cmd.add_argument('test_y', help='path to the test labels', nargs='?', default=None)
    cmd.add_argument('--balance', help='resample the dataset to be balanced', action='store_true')
    cmd.add_argument('--tfidf', help='apply a TF-IDF transform', action='store_true')
    cmd.add_argument('-l', '--lr', default=0.01, help='the learning rate [default=0.01]')
    cmd.add_argument('-b', '--batch-size', default=-1, help='the batch size [default=-1, meaning full]')
    cmd.add_argument('-i', '--iters', default=10, help='the number of iterations [default=10]')
    cmd.set_defaults(func=softmax)

    # hydrus gnb [<args>...] <train_x> <train_y> <test_x> [<test_y>]
    cmd = subcommands.add_parser('gnb', description='Gaussian naive Bayes')
    cmd.add_argument('train_x', help='path to the training set')
    cmd.add_argument('train_y', help='path to the training labels')
    cmd.add_argument('test_x', help='path to the test set')
    cmd.add_argument('test_y', help='path to the test labels', nargs='?', default=None)
    cmd.add_argument('--balance', help='resample the dataset to be balanced', action='store_true')
    cmd.add_argument('--tfidf', help='apply a TF-IDF transform', action='store_true')
    cmd.set_defaults(func=gausian_naive_bayes)

    # hydrus nb [<args>...] <train_x> <train_y> <test_x> [<test_y>]
    cmd = subcommands.add_parser('nb', description='Naive Bayes')
    cmd.add_argument('train_x', help='path to the training set')
    cmd.add_argument('train_y', help='path to the training labels')
    cmd.add_argument('test_x', help='path to the test set')
    cmd.add_argument('test_y', help='path to the test labels', nargs='?', default=None)
    cmd.add_argument('--balance', help='resample the dataset to be balanced', action='store_true')
    cmd.add_argument('--tfidf', help='apply a TF-IDF transform', action='store_true')
    cmd.set_defaults(func=naive_bayes)

    # hydrus preprocess [-a] <train_x> <train_y>
    cmd = subcommands.add_parser('preprocess', description='Inspect the data loader and TF-IDF transformer')
    cmd.add_argument('train_x', help='path to the training set')
    cmd.add_argument('train_y', help='path to the training labels')
    cmd.add_argument('-a', '--all', action='store_true', help='print all data points')
    cmd.set_defaults(func=preprocess)

    args = parser.parse_args()
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
