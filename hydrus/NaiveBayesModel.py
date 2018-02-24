import hydrus.preprocess as hf
import hydrus.postprocess as pp
import html
import re
import nltk
import numpy as np
import pyspark
import math

class Naive_Bayes_Model:

    def __init__(self, ctx):
        '''Initialize a Loader from a SparkContext.
        '''
        self.ctx = ctx

    def no_doc_in_class(self, class_doc, labels):
        """
        This method returns number of documents in a class
        class_doc gives the class name
        labels is a map which contains
        """
        value_list = labels.values()
        count = 0
        for label in value_list:
            if(label == class_doc):
                count = count+1
        return count;

    def fit(self, data, labels):
        """
        This is the main method of the model which accepts preprocessed data and labels associated with it.
        This method returns four parameters
        logprior: this is the prior probability taken log for the classes
        log_likelyhood: This is the likelyhood probabilities for each word and class, taken log
        vocabulary: This represents the vocabulary or all the words from the training data set
        classes_list: This is list of all the classes we want documents classified in
        """
        data_with_counts_only = data.map(lambda x: (x[0], x[1]))
        labels_dictionary = labels.collectAsMap()
        classes_with_words = data_with_counts_only.map(lambda x: ((labels_dictionary[x[0][0]],x[0][1]),x[1])).reduceByKey(lambda x,y: x+y)
        classes_with_word_count = classes_with_words.map(lambda x: (x[0][0], x[1])).reduceByKey(lambda x,y:x+y).collectAsMap()
        classes_list = classes_with_word_count.keys()
        documents_list = labels.collectAsMap().keys()
        vocabulary = data.map(lambda x: (x[0][1],x[1])).reduceByKey(lambda x,y: x+y).collectAsMap().keys()

        n_doc = len(documents_list)
        logprior = {}
        log_likelyhood = {}

        for class_docs in classes_list:

            n_doc = len(documents_list)
            n_doc_class = Naive_Bayes_Model(ctx).no_doc_in_class(class_docs, labels_dictionary)
            words_in_class = classes_with_words.filter(lambda x: x[0][0] == class_docs).map(lambda x: (x[0][1],x[1])).collectAsMap()
            logprior[class_docs] = np.log(n_doc_class/n_doc)
            likelyhood_denominator = len(vocabulary) + classes_with_word_count.get(class_docs)

            #for word in vocabulary:
            #    likelyhood_denominator = likelyhood_denominator + words_in_class.get(word, 0) + 1

            for word in vocabulary:
                count_word_in_class = words_in_class.get(word, 0) + 1
                log_likelyhood_value = np.log(count_word_in_class/likelyhood_denominator)
                log_likelyhood[word,class_docs] = log_likelyhood_value
        self.logprior = logprior
        print('log prior is: ', logprior)
        self.log_likelyhood = log_likelyhood
        self.vocabulary = vocabulary
        self.classes_list = classes_list
        self.classes_with_word_count = classes_with_word_count
        return  self

    def words_in_docement(self, document, test_data):
        """
        This method simply returns count of words in a document
        document: it is the name/id of the document
        test_data: it is the preprocessed test data
        """
        words_list = test_data.filter(lambda x: x[0][0] == document).map(lambda x: (x[0][1], x[1])).collectAsMap().keys()
        return words_list

    def predict(self, test_data):
        """
        This is where we predict the classification of documents.
        It takes probabilities calculated in the train_naive_bayes method as input and outputs probability_doc
        This is a dictionary which has document id as key and its predicted class as value
        """
        test_data_words = test_data.map(lambda x: (x[0][1],x[1])).reduceByKey(lambda x,y: x+y).collectAsMap().keys()
        documents_list = test_data.map(lambda x: (x[0][0],x[1])).reduceByKey(lambda x,y: x+y).collectAsMap().keys()
        probability_doc = {}

        for document in documents_list:
            words = Naive_Bayes_Model(ctx).words_in_docement(document, test_data)
            probabilities = {}

            for _class in self.classes_list:
                naive_bayes_probability = self.logprior[_class]

                for word in words:

                    if word in self.vocabulary:
                        naive_bayes_probability = naive_bayes_probability + self.log_likelyhood[word,_class]
                    else:
                        naive_bayes_probability = naive_bayes_probability + np.log(float(1 / (1+len(self.vocabulary)+self.classes_with_word_count.get(_class))))
                probabilities[naive_bayes_probability] = _class
                print(' class: ', _class+ 'word: ', word+ 'probability: ',naive_bayes_probability)
            print('probabilities are: ',probabilities, 'doc ', document)
            probability_doc[document] = probabilities[max(probabilities.keys())]
        self.probability_doc = probability_doc
        return self

    def score(self, test_label):
        """
        This is the method to calculate accuracy of our model.
        """
        matched = 0
        test_label_map = test_label.collectAsMap();
        test_label_length = test_label.map(lambda x: x[0][0]).distinct().count()
        print('length of test lales distinct is', test_label_length)
        for doc in test_label_map.keys():
            if(test_label_map[doc] == self.probability_doc.get(doc,'filler')):
                matched = matched +1
        accuracy = (float(matched)/test_label_length)*100
        print('label file is', test_label_length)
        return accuracy, self.probability_doc

if __name__ == '__main__':
    """
    This is the main method which gets invoked when Command line argument is passed
    """
    import argparse
    parser = argparse.ArgumentParser(description='Inspect the data loader and TF-IDF transformer')
    parser.add_argument('data_path', help='path to the data file')
    parser.add_argument('label_path', help='path to the label file')
    parser.add_argument('test_data_path', help='path to the test data file')
    parser.add_argument('test_label_path', help='path to the test label file')
    parser.add_argument('-a', '--all', action='store_true', help='print all data points')
    args = parser.parse_args()

    conf = pyspark.SparkConf().setAppName('hydrus-p1-dataloader-test')
    ctx = pyspark.SparkContext(conf=conf)

    data, labels = hf.Loader(ctx).read(args.data_path, args.label_path)
    testData, testLabel = hf.Loader(ctx).read(args.test_data_path, args.test_label_path)
    data = hf.TfIdfTransformer(ctx).fit(data).transform(data)
    testData = hf.TfIdfTransformer(ctx).fit(testData).transform(testData)
    print('labels are',testLabel.collectAsMap())

    accuracy, probability_doc = Naive_Bayes_Model(ctx).fit(data, labels).predict(testData).score(testLabel)
    #pp.print_labels(ctx.parallelize(probability_doc),'/hydrus/output_file.txt')
    print('predicted labels:',probability_doc)
    print('accuracy is:', accuracy)
