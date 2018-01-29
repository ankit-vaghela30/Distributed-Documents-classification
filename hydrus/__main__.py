import argparse
import pyspark

from hydrus.features import Loader, TfIdfTransformer


# TODO: perhaps there is a better way to arrange a SparkContext,
# especially if we are submitting through `spark-submit`.
conf = pyspark.SparkConf().setAppName('hydrus')
ctx = pyspark.SparkContext(conf=conf)


def preprocess(args):
    '''Inspect the output of the data loader and TF-IDF transformer.
    '''
    data, labels = Loader(ctx).read(args.data_path, args.label_path)
    data = TfIdfTransformer(ctx).fit(data).transform(data)

    print('Data sample: ', data.take(1)[0])
    if labels is not None:
        print('Label sample:', labels.take(1)[0])

    if args.all:
        print('All data:')
        def print_rdd(x):
            ((doc, word), count, *features) = x
            print(f'{doc:8}', end='\t')
            print(f'{word:15}', end='\t')
            print(f'{count:3}', end='\t')
            for f in features:
                print(f'{f:.3f}', end='\t')
            print()
        data.foreach(lambda x: print_rdd(x))


def hello(args):
    '''Prints 'hello world' or another string given by `args.string`.

    This is purely an example for how to add new subcommands.
    '''
    if not args.string: args.string = 'hello world'
    print(args.string)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Execute hydrus commands')
    subcommands = parser.add_subparsers()

    # hydrus hello [-s STRING]
    cmd = subcommands.add_parser('hello', description='Prints hello world.')
    cmd.add_argument('-s', '--string', help='print this string instead.')
    cmd.set_defaults(func=hello)

    # hydrus preprocess [-a] data_path label_path
    cmd = subcommands.add_parser('preprocess', description='Inspect the data loader and TF-IDF transformer')
    cmd.add_argument('data_path', help='path to the data file')
    cmd.add_argument('label_path', help='path to the label file')
    cmd.add_argument('-a', '--all', action='store_true', help='print all data points')
    cmd.set_defaults(func=preprocess)

    args = parser.parse_args()
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()
