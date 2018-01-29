import sys


def print_labels(labels, file=sys.stdout):
    '''Prints a label RDD.

    The output is simply each label on a separate line sorted by doc_id.

    Args:
        labels:
            An RDD of the form `(doc_id, label)`.
        file:
            The file to print to. It can be a a file-like object or a string
            or path-like object giving the path to a file.
    '''
    labels = labels.sortByKey()
    labels = labels.map(lambda x: x[1])
    labels = labels.collect()
    if hasattr(file, 'write'):
        print(*labels, sep='\n', file=file)
    else:
        file = open(file, 'w')
        print(*labels, sep='\n', file=file)
        file.close()
