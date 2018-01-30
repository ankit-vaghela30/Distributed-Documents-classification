from . import logistic
from . import math
from . import postprocess
from . import preprocess


def interactive(*args, **kwargs):
    '''Creates a SparkContext for interactive use.
    '''
    import pyspark
    conf = pyspark.SparkConf(*args, **kwargs)
    ctx = pyspark.SparkContext(conf=conf)
    return ctx
