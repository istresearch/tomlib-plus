# coding: utf-8

from nltk.corpus import stopwords
from nltk import tokenize, ISRIStemmer, LancasterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import networkx as nx
import itertools
import pandas
from networkx.readwrite import json_graph
from scipy import spatial
import re
from stop_words import get_stop_words
import snowballstemmer

'''
    Modelled after Adrien Guille and Pavel Soriano's tom_lib package,
    Corpus has the ability to take in documents as strings. It has also been modified to currently 
    support the languages allowed in Python's NLTK package as well as Pashto.
    The Arabic is supported by the snowball light stemmer instead of the
    NLTK ISRIStemmer for Arabic.
'''
class Corpus:

    def __init__(self,
                 text=list(),
                 language=None,
                 n_gram=1,
                 vectorization='tfidf',
                 max_relative_frequency=1.,
                 min_absolute_frequency=0,
                 max_features=2000,
                 sample=None,
                 preprocessed_text = None,
                 stemmed = False):

        self._text = text
        self._language = language
        self._n_gram = n_gram
        self._vectorization = vectorization
        self._max_relative_frequency = max_relative_frequency
        self._min_absolute_frequency = min_absolute_frequency
        self.preprocessed_text = preprocessed_text
        self.max_features = max_features
        
        self.data_frame = pandas.DataFrame({'text': self._text},
                                           columns = ['text'])
        if sample:
            self.data_frame = self.data_frame.sample(frac=0.8)
        self.data_frame.fillna(' ')
        self.size = self.data_frame.count(0)[0]
            
        self.stop_words = []
        if language is not None and language not in ['ar','arabic']:
            self.stop_words = stopwords.words(language)
        
        elif language in ['ar','arabic']:
            self.stop_words = snowballstemmer.stemmer('arabic').get_stop_words()
            
        
        if vectorization == 'tfidf':
            self.vectorizer = TfidfVectorizer(ngram_range=(1, self._n_gram),
                                         max_df=max_relative_frequency,
                                         min_df=min_absolute_frequency,
                                         max_features=self.max_features,
                                         stop_words=self.stop_words)
        elif vectorization == 'tf':
            self.vectorizer = CountVectorizer(ngram_range=(1, self._n_gram),
                                         max_df=max_relative_frequency,
                                         min_df=min_absolute_frequency,
                                         max_features=self.max_features,
                                         stop_words=stop_words)
        else:
            raise ValueError('Unknown vectorization type: %s' % vectorization)
        self.sklearn_vector_space = self.vectorizer.fit_transform(self.preprocess_text())
        self.gensim_vector_space = None
        vocab = self.vectorizer.get_feature_names()
        self.vocabulary = dict([(i, s) for i, s in enumerate(vocab)])

    def preprocess_text(self):
        
        #don't stem over and over again
        if self.preprocessed_text:
            return self._text
        
        #current stemming  capabilities: arabic and english
        elif self._language in ['ar','arabic']:
            return self.arabic_preprocess(self._text)

        elif self._language in ['en','english']:
            return self.english_preprocess(self._text)

        else:
            return self._text

    def arabic_preprocess(self, doc_texts):

        #step 1: clean up special characters
        doc_texts_cleaned = doc_texts

        for i in range(len(doc_texts)):
            try:
                doc_texts_cleaned[i] = re.sub(r'[\u064b-\u065f]','',doc_texts_cleaned[i])
                doc_texts_cleaned[i] = re.sub(r'[^\s\u0621-\u064A\u0660-\u0669]',' ',doc_texts_cleaned[i])
                doc_texts_cleaned[i] = re.sub('\\n',' ',doc_texts_cleaned[i])
                doc_texts_cleaned[i] = re.sub(r'\d+',' ', doc_texts_cleaned[i])
    
            except Error as e:
                print('Error guys: text is ' + str(doc_texts_cleaned[i]))

        #step 2: tokenize and stem text, removing stop words
        final_text = []
        ar_light_stem = snowballstemmer.stemmer('arabic')

        for i in range(len(doc_texts_cleaned)):
    
            words = tokenize.word_tokenize(doc_texts_cleaned[i])
    
            words_to_add = []
            for j in range(len(words)):
                words[j] = ar_light_stem.stemWord(words[j])
    
            final_text.append([word for word in words if word not in self.stop_words])

        #step 3: put tokens back together into documents
        final_strings = self.detokenize(final_text)
        #print(final_strings[0])
        return final_strings
        
    def english_preprocess(self, doc_texts):
        
        doc_texts_cleaned = doc_texts

        #step 1: clean up special characters
        for i in range(len(doc_texts_cleaned)):
            doc_texts_cleaned[i] = doc_texts_cleaned[i].lower()
            doc_texts_cleaned[i] = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ', doc_texts_cleaned[i])
            doc_texts_cleaned[i] = re.sub(r'&[a-z]+;',' ',doc_texts_cleaned[i])
            doc_texts_cleaned[i] = re.sub(r'^rt',' ',doc_texts_cleaned[i])
            doc_texts_cleaned[i] = re.sub(r'[#\d/:\'\"&\.@_;…]',' ',doc_texts_cleaned[i])

        #step 2: tokenize and stem text, removing stop words
        eng_stemmer = LancasterStemmer()

        final_text = []

        for i in range(len(doc_texts_cleaned)):
            words = tokenize.word_tokenize(doc_texts_cleaned[i])
    
            for j in range(len(words)):
                words[j] = eng_stemmer.stem(words[j])
    
            final_text.append([word for word in words if word not in self.stop_words])

        #step 3: put tokens back together into documents
        final_strings = detokenize(final_text)

        return final_strings

    #takes token sets and returns them into combined docs
    def detokenize(self, token_sets):
        final_strings = []

        for text in token_sets:
            string = ''
            for word in text:
                string = word + ' '
            final_strings.append(string)
        return final_strings

    def vector_for_document(self, doc_id):
        vector = self.sklearn_vector_space[doc_id]
        cx = vector.tocoo()
        weights = [0.0] * len(self.vocabulary)
        for row, word_id, weight in itertools.zip_longest(cx.row, cx.col, cx.data):
            weights[word_id] = weight
        return weights

    def word_for_id(self, word_id):
        return self.vocabulary.get(word_id)
    
    def id_for_word(self, word):
        for i, s in self.vocabulary.items():
            if s == word:
                return i
        return -1
    
    def similar_documents(self, doc_id, num_docs):
        doc_weights = self.vector_for_document(doc_id)
        similarities = []
        for a_doc_id in range(self.size):
            if a_doc_id != doc_id:
                similarity = 1.0 - spatial.distance.cosine(doc_weights, self.vector_for_document(a_doc_id))
                similarities.append((a_doc_id, similarity))
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:num_docs]
    
    def collaboration_network(self, doc_ids=None, nx_format=False):
        nx_graph = nx.Graph(name='')
        for doc_id in doc_ids:
            authors = self.author(doc_id)
            for author in authors:
                nx_graph.add_node(author)
            for i in range(0, len(authors)):
                for j in range(i+1, len(authors)):
                    nx_graph.add_edge(authors[i], authors[j])
        bb = nx.betweenness_centrality(nx_graph)
        nx.set_node_attributes(nx_graph, 'betweenness', bb)
        if nx_format:
            return nx_graph
        else:
            return json_graph.node_link_data(nx_graph)

    
