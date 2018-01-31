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
    counts, labels = Loader(ctx).read(args.train, args.labels)
    tfidfs = TfIdfTransformer(ctx).fit(counts).transform(counts)

    print('Data sample: ', counts.take(1)[0])
    if labels is not None:
        print('Label sample:', labels.take(1)[0])

    if args.all:
        print('All data:')
        def print_rdd(x):
            ((doc, word), (count, tfidf)) = x
            print('{doc:8}'.format(locals()), end='\t')
            print('{word:15}'.format(locals()), end='\t')
            print('{count:3}'.format(locals()), end='\t')
            print('{tfidf:3}'.format(locals()), end='\t')
            print()
        data = sounts.join(labels)
        data.foreach(lambda x: print_rdd(x))


def softmax(args):
    '''Perform a logistic regression.
    '''
    train = args.train
    labels = args.labels
    test = args.test
    lr = float(args.lr)
    batch_size = int(args.batch_size)
    iters = int(args.iters)

    ctx = get_context()
    train, labels = hydrus.preprocess.Loader(ctx).read(train, labels)
    train = hydrus.preprocess.TfIdfTransformer(ctx).fit(train).transform(train)
    test = hydrus.preprocess.Loader(ctx).read(test)
    lr = hydrus.logistic.LogisticRegression(ctx)
    lr.fit(train, labels, lr=lr, batch_size=batch_size, max_iter=iters)
    pred = lr.predict(test)
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

    # hydrus softmax <train> <labels> <test>
    cmd = subcommands.add_parser('softmax', description='multinomial logistic regression')
    cmd.add_argument('train', help='path to the training set')
    cmd.add_argument('labels', help='path to the label file')
    cmd.add_argument('test', help='path to the test set')
    cmd.add_argument('-l', '--lr', default=0.01, help='the learning rate [default=0.01]')
    cmd.add_argument('-b', '--batch-size', default=-1, help='the batch size [default=-1, meaning full]')
    cmd.add_argument('-i', '--iters', default=10, help='the number of iterations [default=10]')
    cmd.set_defaults(func=softmax)

    # hydrus preprocess [-a] <train> <labels>
    cmd = subcommands.add_parser('preprocess', description='Inspect the data loader and TF-IDF transformer')
    cmd.add_argument('train', help='path to the training set')
    cmd.add_argument('labels', help='path to the label file')
    cmd.add_argument('-a', '--all', action='store_true', help='print all data points')
    cmd.set_defaults(func=preprocess)

    args = parser.parse_args()
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
