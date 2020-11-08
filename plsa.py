import numpy as np
import math


def normalize(input_matrix):
    """
    Normalizes the rows of a 2d input_matrix so they sum to 1
    """

    row_sums = input_matrix.sum(axis=1)
    try:
        assert (np.count_nonzero(row_sums)==np.shape(row_sums)[0]) # no row should sum to zero
    except Exception:
        raise Exception("Error while normalizing. Row(s) sum to zero")
    new_matrix = input_matrix / row_sums[:, np.newaxis]
    return new_matrix

       
class Corpus(object):

    """
    A collection of documents.
    """

    def __init__(self, documents_path):
        """
        Initialize empty document list.
        """
        self.documents = []
        self.vocabulary = []
        self.likelihoods = []
        self.documents_path = documents_path
        self.term_doc_matrix = None 
        self.document_topic_prob = None  # P(z | d)
        self.topic_word_prob = None  # P(w | z)
        self.topic_prob = None  # P(z | d, w)

        self.number_of_documents = 0
        self.vocabulary_size = 0

    def build_corpus(self):
        """
        Read document, fill in self.documents, a list of list of word
        self.documents = [["the", "day", "is", "nice", "the", ...], [], []...]
        
        Update self.number_of_documents
        """
        # #############################
        # your code here
        # #############################
        with open(self.documents_path) as f:
            lines = []
            for line in f:
                lines = line.split()
                self.documents.append(lines)
        self.number_of_documents = len(self.documents)

    def build_vocabulary(self):
        """
        Construct a list of unique words in the whole corpus. Put it in self.vocabulary
        for example: ["rain", "the", ...]

        Update self.vocabulary_size
        """
        # #############################
        # your code here
        # #############################
        """
        unique_words = set()
        for document in self.documents:
            for word in document:
                unique_words.add(word)
        self.vocabulary = list(unique_words)
        self.vocabulary_size = len(self.vocabulary)
        """
        unique_words = set()
        with open(self.documents_path, 'r') as file:
            for line in file:
                for word in line.split():
                    unique_words.add(word)
        self.vocabulary = list(unique_words)
        self.vocabulary_size = len(self.vocabulary)

    def build_term_doc_matrix(self):
        """
        Construct the term-document matrix where each row represents a document, 
        and each column represents a vocabulary term.

        self.term_doc_matrix[i][j] is the count of term j in document i
        """
        # ############################
        # your code here
        # ############################
        matrix = {}
        for  i in range (0, self.number_of_documents):
            matrix[i] = {}
            for j in range (0, self.vocabulary_size):
                matrix[i][j] = self.documents[i].count(self.vocabulary[j])
        self.term_doc_matrix = matrix
        


    def initialize_randomly(self, number_of_topics):
        """
        Randomly initialize the matrices: document_topic_prob and topic_word_prob
        which hold the probability distributions for P(z | d) and P(w | z): self.document_topic_prob, and self.topic_word_prob

        Don't forget to normalize! 
        HINT: you will find numpy's random matrix useful [https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.random.random.html]
        """
        # ############################
        # your code here
        # ############################
        
        self.document_topic_prob = np.random.random((self.number_of_documents, number_of_topics))
        self.document_topic_prob = normalize(self.document_topic_prob)
        
        self.topic_word_prob = np.random.random((number_of_topics, self.vocabulary_size))
        self.topic_word_prob = normalize(self.topic_word_prob)
        

    def initialize_uniformly(self, number_of_topics):
        """
        Initializes the matrices: self.document_topic_prob and self.topic_word_prob with a uniform 
        probability distribution. This is used for testing purposes.

        DO NOT CHANGE THIS FUNCTION
        """
        self.document_topic_prob = np.ones((self.number_of_documents, number_of_topics))
        self.document_topic_prob = normalize(self.document_topic_prob)

        self.topic_word_prob = np.ones((number_of_topics, len(self.vocabulary)))
        self.topic_word_prob = normalize(self.topic_word_prob)

    def initialize(self, number_of_topics, random=False):
        """ Call the functions to initialize the matrices document_topic_prob and topic_word_prob
        """
        print("Initializing...")

        if random:
            self.initialize_randomly(number_of_topics)
        else:
            self.initialize_uniformly(number_of_topics)

    def expectation_step(self):
        """ The E-step updates P(z | w, d)
        """
        print("E step:")
        
        # ############################
        # your code here
        # ############################
        for d in range(self.number_of_documents):
            for w in range(self.vocabulary_size):
                prob = np.zeros([number_of_topics], dtype = float)
                for z in range(number_of_topics):
                   prob[z] = self.document_topic_prob[d, z] * self.topic_word_prob[z, w]
                sum1 = np.sum(prob)
                if summ1 == 0:
                   sum1 = 1 
                for z in range(number_of_topics):
                   self.topic_prob[d,w,z] = prob[z]/sum1

    def maximization_step(self, number_of_topics):
        """ The M-step updates P(w | z)
        """
        print("M step:")
        
        # update P(w | z)
        for z in range(number_of_topics):
                for w in range(self.vocabulary_size):
                    s = 0
                    for d in range(self.number_of_documents):
                        count = term_doc_matrix[d][w]
                        s = s + count * self.topic_prob[d, w, z]
                    self.topic_word_prob[z][w] = s
                normalize(self.topic_word_prob[z])
        # ############################
        # your code here
        # ############################

        
        # update P(z | d)
        for d in range(self.number_of_documents):
                for z in range(number_of_topics):
                    s = 0
                    for w in range(self.vocabulary_size):
                        count = term_doc_matrix[d][w]
                        s = s + count * self.topic_prob[d, w, z]
                    self.document_topic_prob[d][z] = s
                normalize(self.document_topic_prob[d])

        # ############################
        # your code here
        # ############################


    def calculate_likelihood(self, number_of_topics):
        """ Calculate the current log-likelihood of the model using
        the model's updated probability matrices
        
        Append the calculated log-likelihood to self.likelihoods

        """
        for d in range(self.number_of_documents):
            for z in range(number_of_topics):
                total = 0
                for w in range(self.vocabulary_size):
                    total += self.topic_prob[d, z, w] * self.topic_word_prob[z, w] * self.term_doc_matrix[d][w]
                    total += np.log(total)
        return total
        # ############################
        # your code here
        # ############################
        
        return

    def plsa(self, number_of_topics, max_iter, epsilon):

        """
        Model topics.
        """
        print ("EM iteration begins...")
        
        # build term-doc matrix
        self.build_term_doc_matrix()
        
        # Create the counter arrays.
        
        # P(z | d, w)
        self.topic_prob = np.zeros([self.number_of_documents, number_of_topics, self.vocabulary_size], dtype=np.float)

        # P(z | d) P(w | z)
        self.initialize(number_of_topics, random=True)

        # Run the EM algorithm
        current_likelihood = 0.0

        for iteration in range(max_iter):
            print("Iteration #" + str(iteration + 1) + "...")

            # ############################
            # your code here
            # ############################
            self.expectation_step()
            self.maximization_step(number_of_topics)
            self.calculate_likelihood(number_of_topics)
            current_likelihood = self.likelihoods[len(self.likelihoods) - 1]
            print(current_likelihood)
            if (abs(current_likelihood - self.likelihoods) <= epsilon):
                break
      



def main():
    documents_path = 'data/test.txt'
    corpus = Corpus(documents_path)  # instantiate corpus
    corpus.build_corpus()
    corpus.build_vocabulary()
    print(corpus.vocabulary)
    print("Vocabulary size:" + str(len(corpus.vocabulary)))
    print("Number of documents:" + str(len(corpus.documents)))
    number_of_topics = 2
    max_iterations = 50
    epsilon = 0.001
    corpus.plsa(number_of_topics, max_iterations, epsilon)



if __name__ == '__main__':
    main()
