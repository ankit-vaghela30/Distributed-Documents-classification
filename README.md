# Hydrus
Distributed document classification

Hydrus was a course project for CSCI 8360 Data Science Practicum @ UGA to implement document classification in Spark for large datasets. The catch is that no Spark libraries are permitted, not even Spark's own MLlib. Everything has been implemented from the ground up atop Spark's Resilient Distributed Dataset (RDD) primitives and deployed to Google Cloud Dataproc.


## Quick start

### Run locally

Hydrus is implemented in Python with PySpark, and distributed as a plain-old Python package. Launch it with `python -m`:

```
$ python -m hydrus <command> <args>
```

The following commands are supported:

- `hydrus nb [<args>...] <train_x> <train_y> <test_x> [<test_y>]`
- `hydrus gnb [<args>...] <train_x> <train_y> <test_x> [<test_y>]`
- `hydrus softmax [<args>...] <train_x> <train_y> <test_x> [<test_y>]`
- `hydrus preprocess [-a | --all] <train_x> <train_y>`
- `hydrus info`

The `nb`, `gnb`, and `softmax` commands train a naive Bayes, Gaussian naive Bayes, and softmax (multinomial logistic) regression classifiers respectively. Each of these require a training set and labels and a test set. If labels are given for the test set, the accuracy is printed to stdout, otherwise predictions for the test set are printed.

Each of these classifier commands optionally take the arguments `--balance` to resample the dataset from the label priors and `--tf-idf` to classify on TF-IDF scores instead of raw word counts.

The `preprocess` command prints samples from the word counter, label reader, and TF-IDF transformer. The `--all` option causes the command to output all samples from the input set. This command can be useful when exploring new data or debugging the input pipeline.

The `info` command prints out system info. This can be useful to gain information about a remote Spark cluster.

### Deploy to Spark

The easiest way to submit a Hydrus job to a Spark cluster is to compile the package into an egg and use the thin driver at `scripts/driver.py`:

```shell
# Build the egg
./setup.py bdist_egg

# Submit the job
spark-submit --py-files ./dist/hydrus-*.egg scripts/driver.py <args>

```

For Google Cloud Dataproc, use `gcloud` instead of `spark-submit`:

```shell
# Build the egg
./setup.py bdist_egg

# Submit the job
gcloud dataproc jobs submit pyspark \
	--cluster <cluster> \
	--region <region> \
	--py-files ./dist/hydrus-*.egg \
	scripts/driver.py \
	-- <args>

```


## Internals

Hydrus provides three classifier classes with a Scikit-learn style API: `hydrus.naive_bayes.NaiveBayes`, `hydrus.naive_bayes.GaussianNaiveBayes`, and `hydrus.logistic.LogisticRegression`. These operate RDDs of the form `((id, feature), value)` where `id` uniquely identifies each instance, `feature` identifies a feature available on the instance, and `value` is a numeric value for that feature on that instance. Training labels and prediction outputs are RDDs of the form `(id, label)`.

The `hydrus.preprocess` package provides a `Loader` class to read in word counts and labels from text data. The data files should contain one document per line, and label files should contain `,` separated labels for the document on the same line. This loader class is specific to the UGA assignment and ignores any label not ending with the string `'CAT'`. This package also includes a `TfIdfTransformer` class to transform word counts into TF-IDF values.

A key abstraction for the softmax/logistic classifier is the `RddTensor` class in the `hydrus.math` package. This library is responsible for mapping between the relational algebra concepts of RDDs and linear algebra concepts common to machine learning. In particular, this library implements matrix multiplication as a join-map-reduce operation and is especially well suited for sparse matrices which are common to text processing.


## Reproducibility

The best results we achieved were with a pure Python prototype. We are currently having trouble reproducing the results with the Spark models. See [#36] for the discussion of the issue and [#37] for the most recent attempts to improve the accuracy.

[#36]: https://github.com/dsp-uga/hydrus/issues/36
[#37]: https://github.com/dsp-uga/hydrus/pull/37
