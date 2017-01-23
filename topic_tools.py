'''
    This class is meant to take in a json dataset, fetch the
    text from all of the text  field of data that are of the given type (or
    all types), and output all of the topics and other metadata necessary
    to then be able to plot topic model data in a UI.
'''
import utils
import stats
from corpus import Corpus
import json
import itertools
import pandas as pd
#from nltk.classify.textcat import TextCat
from sklearn.decomposition import NMF
from scipy.sparse import coo_matrix, vstack
import numpy as np
from abc import ABCMeta, abstractmethod
from sklearn.decomposition import NMF

class TopicModel:

    #initialize topic model tool
    def __init__(self, dataset=[], type = ['telegram_stream_message', 'fbgraph_post', 'tweet_traptor'],
                 language = 'english', n_gram = 1,
                 vectorization = 'tfidf', max_relative_frequency=1.,
                 min_absolute_frequency=0,
                 max_features=2000,
                 sample = None,
                 preprocessed_text=None):

        self.document_topic_matrix = None  # document x topic matrix
        self.topic_word_matrix = None  # topic x word matrix
        self.nb_topics = None  # a scalar value > 1
        self.language = language
        self.dataset = dataset
        self.sample = sample
        self.type = type
        self.n_gram= n_gram
        self.max_relative_frequency = max_relative_frequency
        self.min_absolute_frequency = min_absolute_frequency
        self.max_features = max_features
        self.preprocessed_text = preprocessed_text
        
        if self.n_gram < 1:
            raise ValueError('Invalid n gram value: %d' % n_gram)
        
        self.vectorization = vectorization
        
        

        #if type was not passed, then all three available data types will be included
        for t in self.type:
            if t.strip() not in ['telegram_stream_message', 'fbgraph_post', 'tweet_traptor']:
                raise InvalidTypeError
        
        #narrow down dataset to the types available
        self.dataset = [datum for datum in self.dataset if datum['_type'] in self.type]
        
        #get text from each item using recursive method in utils package
        text = []
        
        for datum in self.dataset:
            doc = utils.get_item_recur('body', datum)
            if not doc:
                doc = utils.get_item_recur('text', datum)
                        
            text.append(str(doc))

        #try getting language data using same method.
        #otherwise use nltk's textcat
        lang = []
        tc = TextCat()
        
        for i in range(len(self.dataset)):

            fetched_language = utils.get_item_recur('lang', self.dataset[i])

            #if language is None:
            #    fetched_language = tc.guess_language(text[i])
            #
            lang.append(fetched_language)

        #provide easy-to-use dataframe of relevant data, this can be output as json
        #easily for visualization use.
        self.dataframe = pd.DataFrame({'text': text,\
                                   'lang': lang,\
                                   'type': [datum['_type'] for datum in self.dataset]},\
                                   columns = ['text', 'lang', 'type'])
                
        self.corpus = Corpus(self.dataframe.text,
                             language = self.language,
                             n_gram=self.n_gram,
                             max_relative_frequency=self.max_relative_frequency,
                             min_absolute_frequency=self.min_absolute_frequency,
                             max_features=self.max_features,
                             sample = self.sample,
                             preprocessed_text = self.preprocessed_text)
            
        self.preprocessed_text = self.corpus.preprocessed_text


    @abstractmethod
    def infer_topics(self, num_topics=10, **kwargs):
        pass

    """
    Implements Greene metric to compute the optimal number of topics. Taken from How Many Topics?
    Stability Analysis for Topic Models from Greene et al. 2014.
    :param step:
    :param min_num_topics: Minimum number of topics to test
    :param max_num_topics: Maximum number of topics to test
    :param top_n_words: Top n words for topic to use
    :param tao: Number of sampled models to build
    :return: A list of len (max_num_topics - min_num_topics) with the stability of each tested k
    """
    def greene_metric(self, min_num_topics=10, max_num_topics=50, top_n_words=10, tao=10):
        stability = []
        # Build reference topic model
        # Generate tao topic models with tao samples of the corpus
        for k in range(min_num_topics, max_num_topics + 1):
            self.infer_topics(k)
            reference_rank = [list(zip(*self.top_words(i, top_n_words)))[0] for i in range(k)]
            agreement_score_list = []
            for t in range(tao):
                '''
                tao_corpus = Corpus(text=self.corpus._text,
                                    language=self.corpus._language,
                                    n_gram=self.corpus._n_gram,
                                    vectorization=self.corpus._vectorization,
                                    max_relative_frequency=self.corpus._max_relative_frequency,
                                    min_absolute_frequency=self.corpus._min_absolute_frequency,
                                    sample=True)
                '''
                tao_model = type(self)(dataset=self.dataset,
                                       n_gram=self.corpus._n_gram,
                                       vectorization=self.corpus._vectorization,
                                       max_relative_frequency=self.corpus._max_relative_frequency,
                                       min_absolute_frequency=self.corpus._min_absolute_frequency,
                                       language = self.language,
                                       sample=True)
                tao_model.infer_topics(k)
                tao_rank = [next(zip(*tao_model.top_words(i, top_n_words))) for i in range(k)]
                agreement_score_list.append(stats.agreement_score(reference_rank, tao_rank))
            stability.append(np.mean(agreement_score_list))
        return stability

    """
    Implements Arun metric to estimate the optimal number of topics:
    Arun, R., V. Suresh, C. V. Madhavan, and M. N. Murthy
    On finding the natural number of topics with latent dirichlet allocation: Some observations.
    In PAKDD (2010), pp. 391–402.
    :param min_num_topics: Minimum number of topics to test
    :param max_num_topics: Maximum number of topics to test
    :param iterations: Number of iterations per value of k
    :return: A list of len (max_num_topics - min_num_topics) with the average symmetric KL divergence for each k
    """
    def arun_metric(self, min_num_topics=10, max_num_topics=50, iterations=10):
        kl_matrix = []
        for j in range(iterations):
            kl_list = []
            l = np.array([sum(self.corpus.vector_for_document(doc_id)) for doc_id in range(self.corpus.size)])
            norm = np.linalg.norm(l)
            for i in range(min_num_topics, max_num_topics + 1):
                self.infer_topics(i)
                c_m1 = np.linalg.svd(self.topic_word_matrix.todense(), compute_uv=False)
                c_m2 = l.dot(self.document_topic_matrix.todense())
                c_m2 += 0.0001
                c_m2 /= norm
                kl_list.append(stats.symmetric_kl(c_m1.tolist(), c_m2.tolist()[0]))
                kl_matrix.append(kl_list)
                ouput = np.array(kl_matrix)
        return ouput.mean(axis=0)

    def top_words(self, topic_id, num_words):
        vector = self.topic_word_matrix[topic_id]
        cx = vector.tocoo()
        weighted_words = [()] * len(self.corpus.vocabulary)
        for row, word_id, weight in itertools.zip_longest(cx.row, cx.col, cx.data):
            weighted_words[word_id] = (self.corpus.word_for_id(word_id), weight)
        weighted_words.sort(key=lambda x: x[1], reverse=True)
        return weighted_words[:num_words]

    def print_topics(self, num_words=10, sort_by_freq=''):
        frequency = self.topics_frequency()
        topic_list = []
        for topic_id in range(self.nb_topics):
            word_list = []
            for weighted_word in self.top_words(topic_id, num_words):
                word_list.append(weighted_word[0])
            topic_list.append((topic_id, frequency[topic_id], word_list))
        if sort_by_freq == 'asc':
            topic_list.sort(key=lambda x: x[1], reverse=False)
        elif sort_by_freq == 'desc':
            topic_list.sort(key=lambda x: x[1], reverse=True)
        for topic_id, frequency, topic_desc in topic_list:
            print('topic %d\t%f\t%s' % (topic_id, frequency, ' '.join(topic_desc)))

    def top_words(self, topic_id, num_words):
        vector = self.topic_word_matrix[topic_id]
        cx = vector.tocoo()
        weighted_words = [()] * len(self.corpus.vocabulary)
        for row, word_id, weight in itertools.zip_longest(cx.row, cx.col, cx.data):
            weighted_words[word_id] = (self.corpus.word_for_id(word_id), weight)
        weighted_words.sort(key=lambda x: x[1], reverse=True)
        return weighted_words[:num_words]

    def word_distribution_for_topic(self, topic_id):
        vector = self.topic_word_matrix[topic_id].toarray()
        return vector[0]

    def topic_distribution_for_document(self, doc_id):
        vector = self.document_topic_matrix[doc_id].toarray()
        return vector[0]

    def topic_distribution_for_word(self, word_id):
        vector = self.topic_word_matrix[:, word_id].toarray()
        return vector.T[0]

    def topic_distribution_for_author(self, author_name):
        all_weights = []
        for document_id in self.corpus.documents_by_author(author_name):
            all_weights.append(self.topic_distribution_for_document(document_id))
        output = np.array(all_weights)
        return output.mean(axis=0)

    def most_likely_topic_for_document(self, doc_id):
        weights = list(self.topic_distribution_for_document(doc_id))
        return weights.index(max(weights))

    def topic_frequency(self, topic, date=None):
        return self.topics_frequency(date=date)[topic]

    def topics_frequency(self, date=None):
        frequency = np.zeros(self.nb_topics)
        if date is None:
            ids = range(self.corpus.size)
        else:
            ids = self.corpus.doc_ids(date)
        for i in ids:
            topic = self.most_likely_topic_for_document(i)
            frequency[topic] += 1.0 / len(ids)
        return frequency

    def documents_for_topic(self, topic_id):
        doc_ids = []
        for doc_id in range(self.corpus.size):
            most_likely_topic = self.most_likely_topic_for_document(doc_id)
            if most_likely_topic == topic_id:
                doc_ids.append(doc_id)
        return doc_ids

    def documents_per_topic(self):
        topic_associations = {}
        for i in range(self.corpus.size):
            topic_id = self.most_likely_topic_for_document(i)
            if topic_associations.get(topic_id):
                documents = topic_associations[topic_id]
                documents.append(i)
                topic_associations[topic_id] = documents
            else:
                documents = [i]
                topic_associations[topic_id] = documents
        return topic_associations

    def affiliation_repartition(self, topic_id):
        counts = {}
        doc_ids = self.documents_for_topic(topic_id)
        for i in doc_ids:
            affiliations = set(self.corpus.affiliation(i))
            for affiliation in affiliations:
                if counts.get(affiliation) is not None:
                    count = counts[affiliation] + 1
                    counts[affiliation] = count
                else:
                    counts[affiliation] = 1
        tuples = []
        for affiliation, count in counts.items():
            tuples.append((affiliation, count))
        tuples.sort(key=lambda x: x[1], reverse=True)
        return tuples


class LatentDirichletAllocation(TopicModel):
    def infer_topics(self, num_topics=10, algorithm='variational', **kwargs):
        self.nb_topics = num_topics
        lda_model = None
        topic_document = None
        if algorithm == 'variational':
            lda_model = LDA(n_topics=num_topics, learning_method='batch')
            topic_document = lda_model.fit_transform(self.corpus.sklearn_vector_space)
        elif algorithm == 'gibbs':
            lda_model = lda.LDA(n_topics=num_topics, n_iter=500)
            topic_document = lda_model.fit_transform(self.corpus.sklearn_vector_space)
        else:
            raise ValueError("algorithm must be either 'variational' or 'gibbs', got '%s'" % algorithm)
        self.topic_word_matrix = []
        self.document_topic_matrix = []
        vocabulary_size = len(self.corpus.vocabulary)
        row = []
        col = []
        data = []
        for topic_idx, topic in enumerate(lda_model.components_):
            for i in range(vocabulary_size):
                row.append(topic_idx)
                col.append(i)
                data.append(topic[i])
        self.topic_word_matrix = coo_matrix((data, (row, col)), shape=(self.nb_topics, len(self.corpus.vocabulary))).tocsr()
        row = []
        col = []
        data = []
        doc_count = 0
        for doc in topic_document:
            topic_count = 0
            for topic_weight in doc:
                row.append(doc_count)
                col.append(topic_count)
                data.append(topic_weight)
                topic_count += 1
                doc_count += 1
                self.document_topic_matrix = coo_matrix((data, (row, col)), shape=(self.corpus.size, self.nb_topics)).tocsr()


class NonNegativeMatrixFactorization(TopicModel):
    def infer_topics(self, num_topics=10, **kwargs):
        self.nb_topics = num_topics
        nmf = NMF(n_components=num_topics)
        topic_document = nmf.fit_transform(self.corpus.sklearn_vector_space)
        self.topic_word_matrix = []
        self.document_topic_matrix = []
        vocabulary_size = len(self.corpus.vocabulary)
        row = []
        col = []
        data = []
        for topic_idx, topic in enumerate(nmf.components_):
            for i in range(vocabulary_size):
                row.append(topic_idx)
                col.append(i)
                data.append(topic[i])
        self.topic_word_matrix = coo_matrix((data, (row, col)), shape=(self.nb_topics, len(self.corpus.vocabulary))).tocsr()
        row = []
        col = []
        data = []
        doc_count = 0
        for doc in topic_document:
            topic_count = 0
            for topic_weight in doc:
                row.append(doc_count)
                col.append(topic_count)
                data.append(topic_weight)
                topic_count += 1
                doc_count += 1
        self.document_topic_matrix = coo_matrix((data, (row, col))).tocsr()
        #removed this part from the coo_matrix above: , shape=(self.corpus.size, self.nb_topics)

class InvalidTypeError(Exception):
    
    def __init__(self):
        print('Invalid type requested. Valid types include \'tweet_traptor\', \
              \'fbgraph_post\', and \'telegram_stream_message\' and must be entered in list format.')
