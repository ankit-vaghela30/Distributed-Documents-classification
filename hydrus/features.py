import html
import re

import nltk
import numpy as np
import pyspark


class Tokenizer:
    '''The default tokenizer, follows the NLTK API.

    This currently implements a regular expression tokenizer.

    If we were allowed to, I'd prefer NLTK's `word_tokenize` function.
    '''

    def __init__(self):
        '''Initialize the Tokenizer.
        '''
        self.word_pattern = re.compile('[\w]+')

    def tokenize(self, text):
        '''Transforms `text` into a list of tokens.
        '''
        text = html.unescape(text)
        return self.word_pattern.findall(text)


class Preprocessor:
    '''The default preprocessor, pretends to be a function.

    It lowercases words, strips leading and trailing punctuation, stems, and
    replaces stopwords with the empty string.
    '''

    def __init__(self, language='english'):
        '''Initialize the Preprocessor.

        Args:
            language:
                The language to use for stemming and stopwords.
            punc:
                The punctuation to strip.
        '''
        self.stemmer = nltk.stem.snowball.SnowballStemmer(language='english')
        self.stopwords = nltk.corpus.stopwords.words(language)

    def __call__(self, text):
        '''Apply the preprocessor to some string.
        '''
        text = text.lower()
        if len(text) == 1: return ''
        if text in self.stopwords: return ''
        if text.isnumeric(): return ''
        text = self.stemmer.stem(text)
        return text


class Loader:
    '''A dataloader for bag-of-words RDDs.

    This class can read text files and optionally associated label files.
    The returned data RDDs have the shape `((doc_id, word) count)` where
    `doc_id` is a unique ID for each document, `word` is a word that appears
    at least once in the daocument, and `count` is the number of occurences.
    Note that we use both `doc_id` and `word` as a compound key.

    The returned label RDDs have the shape `(doc_id, label)` where each
    document has exactly one label.

    The data files are formatted such that each line contains a separate
    document. The label files are formatted such that each line contains a
    comma separated list of labels for that document. We ignore all labels
    except those which end in 'CAT'. If a label file is specified and a
    document does not have a 'CAT' label, it is ignored. On the other hand, If
    a label file is specified and a document has multiple 'CAT' labels, the
    document is duplicated under both labels. If no label file is given,
    documents are neither ignored nor duplicated.
    '''

    def __init__(self, ctx):
        '''Initialize a Loader from a SparkContext.
        '''
        self.ctx = ctx

    def read(self, data_path, label_path=None, tokenizer=None, preprocess=None):
        '''Read a data file and (optionally) a label file.

        Args:
            data_path:
                A path to the data file. Can be anything accepted by
                `SparkContext.textFile` including a Google storage URL.
            label_path:
                A path to the label file. Can be anything accepted by
                `SparkContext.textFile` including a Google storage URL.
            tokenizer:
                The tokenizer to split the documents. This is any object with
                a `tokenize` method, e.g. any of the NLTK tokenizers. The
                default is a `hydrus.loader.Tokenizer`.
            preprocess:
                A function to preprocess words. The function may return the
                empty string to remove the word from the document. The default
                is a `hydrus.loader.Preprocessor`.

        Returns:
            A pair of RDDs `(data, label)`.
            If `label_path` is None, the label RDD is None
        '''
        # Create an RDD of all documents keyed by document ID.
        # We cast the doc_id to str to be easier to work with labels.
        data = self.ctx.textFile(data_path)           # (full_text)
        data = data.zipWithUniqueId()                 # (full_text, doc_id)
        data = data.map(lambda x: (str(x[1]), x[0]))  # (doc_id, full_text)

        # Create an RDD of labels keyed by document ID.
        if label_path is not None:
            labels = self.ctx.textFile(label_path)                  # (label_list)
            labels = labels.zipWithUniqueId()                       # (label_list, doc_id)
            labels = labels.map(lambda x: (str(x[1]), x[0]))        # (doc_id, label_list)
            labels = labels.flatMapValues(lambda s: s.split(','))   # (doc_id, label)
            labels = labels.filter(lambda x: x[1].endswith('CAT'))  # (doc_id, label)
        else:
            labels = None

        # Duplicate or ignore documents depending on the labels.
        if labels is not None:
            data = labels.join(data)                                   # (doc_id, (label, full_text))
            data = data.map(lambda x: (f'{x[0]}_{x[1][0]}', x[1][1]))  # (doc_id+, full_text)
            labels = labels.map(lambda x: (f'{x[0]}_{x[1]}', x[1]))    # (doc_id+, label)

        # Create an RDD of preprocessed words keyed by document ID.
        # Words appear once for each time they appear in the document.
        if tokenizer is None: tokenizer = Tokenizer()
        if preprocess is None: preprocess = Preprocessor(language='english')
        tokenize = tokenizer.tokenize
        data = data.flatMapValues(lambda doc: tokenize(doc))  # (doc_id, word)
        data = data.mapValues(preprocess)                     # (doc_id, word)
        data = data.filter(lambda x: len(x[1]) > 0)           # (doc_id, word)

        # Add a simple word count feature to the data.
        data = data.map(lambda x: ((x[0], x[1]), 1))   # ((doc_id, word), count)
        data = data.reduceByKey(lambda a, b: a + b)    # ((doc_id, word), count)

        return data, labels


class TfIdfTransformer:
    '''A transformer which adds TF-IDF features to a simple bag-of-words RDD.
    '''

    def __init__(self, ctx):
        '''Initialize a TfIdfTransformer from a SparkContext.
        '''
        self.ctx = ctx
        self._n_docs = None

    def fit(self, data):
        '''Fit the TF-IDF transform to the training data.
        '''
        # Get the number of documents containing each word.
        # We collect to a dict to be used in the TF map function.
        n_docs = data                        # ((doc_id, word), count)
        n_docs = n_docs.map(lambda x: x[0])  # (doc_id, word)
        n_docs = n_docs.countByValue()       # {word: n_docs}
        self._n_docs = n_docs
        return self

    def transform(self, data):
        '''Adds three features to a bag-of-words RDD: TF, IDF, and TF*IDF.

        Currently this implements the "standard" TF-IDF.

        The TF component is just the frequency that the word occurs within the
        document, i.e. the count divided by the doc length.

        The IDF component is the log of the smoothed inverse document
        frequency, i.e. the number of docs divided by one plus the number of
        docs containing the word. The smoothing avoids divide-by-zero issues.

        Args:
            data:
                An RDD with the compound key `(doc_id, word)` where the first
                feature is the number of occurences of that word in that doc.
                The RDD may contain additional features.

        Returns:
            An RDD like the input, but with the TF, IDF, and TF*IDF features
            appended to the end.
        '''
        # Get the length of each document.
        # We collect to a dict to be used in the TF map function.
        lengths = data                                     # ((doc_id, word), count)
        lengths = lengths.map(lambda x: (x[0][0], x[1]))   # (doc_id, length)
        lengths = lengths.reduceByKey(lambda a, b: a + b)  # (doc_id, length)
        lengths = lengths.collectAsMap()                   # {doc_id: length}

        n_docs_total = len(lengths)
        n_docs = self._n_docs  # {word: n_docs}

        def tf_idf(x):
            ((doc, word), count, *others) = x
            n_docs_word = n_docs.get(word, 0) + 1  # +1 smoothing prevents divide by zero
            tf = count / lengths[doc]
            idf = np.log(n_docs_total / n_docs_word)
            tf_idf = tf * idf
            return ((doc, word), count, *others, tf, idf, tf_idf)

        data = data.map(tf_idf, preservesPartitioning=True)
        return data


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Inspect the data loader and TF-IDF transformer')
    parser.add_argument('data_path', help='path to the data file')
    parser.add_argument('label_path', help='path to the label file')
    parser.add_argument('-a', '--all', action='store_true', help='print all data points')
    args = parser.parse_args()

    conf = pyspark.SparkConf().setAppName('hydrus-p1-dataloader-test')
    ctx = pyspark.SparkContext(conf=conf)

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
