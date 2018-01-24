import nltk
import numpy as np
import pyspark


class Tokenizer:
    '''The default tokenizer, follows the NLTK API.

    It simply calls out to NLTK's `word_tokenize` function.
    '''

    def __init__(self, **kwargs):
        '''Initialize the Tokenizer.

        Args:
            **kwargs:
                Forwarded to the `nltk.tokenize.word_tokenize` function.
        '''
        self.kwargs = kwargs

    def tokenize(self, text):
        '''Transforms `text` into a list of tokens.'''
        return nltk.tokenize.word_tokenize(text, **self.kwargs)


class Preprocessor:
    '''The default preprocessor, pretends to be a function.

    It lowercases words, strips leading and trailing punctuation, stems, and
    replaces stopwords with the empty string.
    '''

    def __init__(self, language='english', punc=".,:;'!?&+!()-[]{}\<>/?@#$%^*_~="):
        '''Initialize the Preprocessor.

        Args:
            language:
                The language to use for stemming and stopwords.
            punc:
                The punctuation to strip.
        '''
        self.stemmer = nltk.stem.snowball.SnowballStemmer(language='english')
        self.stopwords = nltk.corpus.stopwords.words(language)
        self.punc = punc

    def __call__(self, text):
        '''Apply the preprocessor to some string.'''
        text = text.lower()
        text = text.strip(self.punc)
        text = self.stemmer.stem(text)
        if text in self.stopwords:
            text = ''
        return text


class Loader:
    '''A factory for constructing a bag-of-words RDD from a text dataset.

    Loaders construct an RDD with a compound key `(doc_id, word)` where
    `doc_id` is a unique identifier for each document and `word` is a
    preprocessed word from the document.

    Initially, the RDD contains a single feature, the number of of occurences
    of the keyed word in the keyed document. Additional features can be added
    through methods like `tf` and `tf_idf`.

    Every feature is given a name. The `index` method of the loader maps these
    feature names to the index of that feature within the RDD row. By default,
    the initial feature is called "count" and additional features are named
    after the method which created them.

    The Loader will also load a label RDD if a label file is provided.
    The label RDD maps each `doc_id` to a string label.

    The data files are formatted such that each line contains a separate
    document. The label files are formatted such that each line contains a
    comma separated list of labels for that document. We ignore all labels
    those which end in 'CAT'. If a label file is specified and a document
    does not have a 'CAT' label, it is ignored. If a label file is specified
    and a document has multiple 'CAT' labels, the document is duplicated under
    both labels. If no label file is given, documents are neither ignored nor
    duplicated.

    The format of the RDD created by a loader is given in a human form through
    the `__str__` method, and thus can be checked by printing the loader.
    '''

    def __init__(self, ctx):
        '''Construct a Loader from a SparkContext.'''
        self.ctx = ctx
        self._features = []
        self._n_docs = None
        self._lengths = None
        self._data = None
        self._labels = None

    def read(self, data_path, label_path=None, name='count',
            tokenizer=None, preprocess=None):
        '''Read a data file and (optionally) a label file.

        This will reset the loader and load new data from the `data_path`.

        The loader will initially build a simple count feature
        with the given name.

        Args:
            data_path:
                A path to the data file. Can be anything accepted by
                `SparkContext.textFile` including a Google storage URL.
            label_path:
                A path to the label file. Can be anything accepted by
                `SparkContext.textFile` including a Google storage URL.
            name:
                The name of the initial wordcount feature.
            tokenizer:
                The tokenizer to split the documents. This is any object with
                a `tokenize` method, e.g. any of the NLTK tokenizers. The
                default is a `hydrus.loader.Tokenizer`.
            preprocess:
                A function to preprocess words. The function may return the
                empty string to remove the word from the document.
        '''
        # Create an RDD of all documents keyed by document ID.
        # We cast the doc_id to str to be easier to work with labels.
        docs = self.ctx.textFile(data_path)           # (full_text)
        docs = docs.zipWithUniqueId()                 # (full_text, doc_id)
        docs = docs.map(lambda x: (str(x[1]), x[0]))  # (doc_id, full_text)

        # Create an RDD of labels keyed by document ID.
        if label_path is not None:
            labels = self.ctx.textFile(label_path)                  # (label_list)
            labels = labels.zipWithUniqueId()                       # (label_list, doc_id)
            labels = labels.map(lambda x: (str(x[1]), x[0]))        # (doc_id, label_list)
            labels = labels.flatMapValues(lambda s: s.split(','))   # (doc_id, label)
            labels = labels.filter(lambda x: x[1].endswith('CAT'))  # (doc_id, label)
        else:
            labels = None

        # Duplicate or ignore as needed
        if labels is not None:
            docs = labels.join(docs)                                   # (doc_id, (label, full_text))
            docs = docs.map(lambda x: (f'{x[0]}_{x[1][0]}', x[1][1]))  # (doc_id+, full_text)
            labels = labels.map(lambda x: (f'{x[0]}_{x[1]}', x[1]))    # (doc_id+, label)

        # Create a dict mapping documents to their lengths.
        lengths = docs.map(lambda x: (x[0], len(x[1])))  # (doc_id, length)
        lengths = lengths.collectAsMap()                 # {doc_id: length}

        # Create an RDD of preprocessed words keyed by document ID.
        # Words appear once for each time they appear in the document.
        if tokenizer is None: tokenizer = Tokenizer(language='english')
        if preprocess is None: preprocess = Preprocessor(language='english')
        tokenize = tokenizer.tokenize
        words = docs.flatMapValues(lambda doc: tokenize(doc))  # (doc_id, word)
        words = words.mapValues(str.lower)                     # (doc_id, word)
        words = words.mapValues(preprocess)                    # (doc_id, word)
        words = words.filter(lambda x: len(x[1]) > 0)          # (doc_id, word)

        # Create an RDD mapping document-word pairs to counts and frequencies.
        # This is the RDD that we will give to the user.
        # We will append additional features as requested.
        data = words.map(lambda x: ((x[0], x[1]), 1))  # ((doc_id, word), count)
        data = data.reduceByKey(lambda a, b: a + b)    # ((doc_id, word), count)

        # Create a dict mapping each word to the number of docs in which it appears.
        # This is a major performance bottleneck.
        n_docs = words.distinct()                        # (doc_id, word)
        n_docs = n_docs.map(lambda x: (x[1], 1))         # (word, n_doc)
        n_docs = n_docs.reduceByKey(lambda a, b: a + b)  # (word, n_doc)
        n_docs = n_docs.collectAsMap()                   # {word: n_doc}

        self._features = [name]
        self._n_docs = n_docs
        self._lengths = lengths
        self._data = data
        self._labels = labels
        return self

    def tf(self, name='tf'):
        '''Adds a frequency feature with the given name.'''
        lengths = self._lengths
        def compute_freq(x):
            key = x[0]
            doc = key[0]
            count = x[1]
            length = lengths[doc]
            return (*x, count/length)
        self._data = self._data.map(compute_freq, preservesPartitioning=True)
        self._features.append(name)
        return self

    def idf(self, name='idf'):
        '''Adds a standard IDF feature with the given name.'''
        n_docs_total = len(self._lengths)
        n_docs = self._n_docs
        def compute_idf(x):
            key = x[0]
            word = key[1]
            n_docs_w = n_docs[word]
            return (*x, np.log(n_docs_total / n_docs_w))
        self._data = self._data.map(compute_idf, preservesPartitioning=True)
        self._features.append(name)
        return self

    def tf_idf(self, name='tf_idf', tf_name='tf', idf_name='idf'):
        '''Adds a TF-IDF feature with the given name.

        If `tf_name` refers to an existing feature, it is used as the TF
        component. Otherwise a new standard TF feature is added with that name
        and used.

        Likewise, if `idf_name` refers to an existing feature, it is used as
        the IDF component. Otherwise a new standard IDF feature is added with
        that name and used.
        '''
        if tf_name not in self._features: self.tf(tf_name)
        if idf_name not in self._features: self.idf(idf_name)
        i = self.index(tf_name)
        j = self.index(idf_name)
        def compute_tf_idf(x):
            tf = x[i]
            idf = x[j]
            return (*x, tf*idf)
        self._data = self._data.map(compute_tf_idf, preservesPartitioning=True)
        self._features.append(name)
        return self

    def load(self):
        '''Returns the data and label RDDs.

        The label RDD is None if the `label_path` was None at the read step.
        '''
        return self._data, self._labels

    def index(self, name):
        '''Returns the index of the feature with the given name.

        If multiple features were added with the same name, this method
        returns the index of the most recently created choice.

        Note that index 0 always belongs to the key.
        '''
        l = len(self._features)
        r = self._features[-1::-1] # reverse of the list

        # This is intentionally off-by-one. `self._features` contains one
        # fewer elements than the number of elements in a data point.
        return l - r.index(name)

    def __str__(self):
        '''Returns a human-readable description of the data RDD.'''
        s = f'Loader: ((doc_id, word)'
        for f in self._features:
            s += ', ' + f
        s += ')'
        return s


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Test the data loader')
    parser.add_argument('data_path', nargs='?', default='./X_train_vsmall.txt', help='path to the data file')
    parser.add_argument('label_path', nargs='?', default=None, help='path to the label file')
    args = parser.parse_args()

    conf = pyspark.SparkConf().setAppName('hydrus-p1-dataloader-test')
    ctx = pyspark.SparkContext(conf=conf)

    loader = Loader(ctx) \
        .read(args.data_path, args.label_path) \
        .tf_idf()

    data, labels = loader.load()

    print(loader)
    print('Example:', data.take(1)[0])
    if labels is not None:
        print('Label Example:', labels.take(1)[0])
