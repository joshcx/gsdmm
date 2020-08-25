from numpy.random import multinomial
from numpy import log, exp
from numpy import argmax
import json
import numpy as np
from tqdm import tqdm

# to perform a deep copy
# of the list of dictionaries
import copy
from sklearn.feature_extraction.text import CountVectorizer


class MovieGroupProcess:
    def __init__(self, K=8, alpha=0.1, beta=0.1, n_iters=30):
        """
        A MovieGroupProcess is a conceptual model introduced by Yin and Wang 2014 to
        describe their Gibbs sampling algorithm for a Dirichlet Mixture Model for the
        clustering short text documents.
        Reference: http://dbgroup.cs.tsinghua.edu.cn/wangjy/papers/KDD14-GSDMM.pdf

        Imagine a professor is leading a film class. At the start of the class, the students
        are randomly assigned to K tables. Before class begins, the students make lists of
        their favorite films. The teacher reads the role n_iters times. When
        a student is called, the student must select a new table satisfying either:
            1) The new table has more students than the current table.
        OR
            2) The new table has students with similar lists of favorite movies.

        :param K: int
            Upper bound on the number of possible clusters. Typically many fewer
        :param alpha: float between 0 and 1
            Alpha controls the probability that a student will join a table that is currently empty
            When alpha is 0, no one will join an empty table.
        :param beta: float between 0 and 1
            Beta controls the student's affinity for other students with similar interests. A low beta means
            that students desire to sit with students of similar interests. A high beta means they are less
            concerned with affinity and are more influenced by the popularity of a table
        :param n_iters:
        """
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.n_iters = n_iters

        # slots for computed variables
        self.number_docs = None
        self.vocab_size = None
        self.cluster_doc_count = [0 for _ in range(K)]
        self.cluster_word_count = [0 for _ in range(K)]
        self.cluster_word_distribution = [{} for i in range(K)]

        # newly added variables
        self.topic_term_list = []
        self.vocab = []
        self.topic_map = {}
        # needs be updated later as
        # vocab size is now zero
        self.topic_term_matrix = None
        self.doc_topic_matrix = None
        self.doc_term_matrix = None

    @staticmethod
    def from_data(
        K,
        alpha,
        beta,
        D,
        vocab_size,
        cluster_doc_count,
        cluster_word_count,
        cluster_word_distribution,
    ):
        """
        Reconstitute a MovieGroupProcess from previously fit data
        :param K:
        :param alpha:
        :param beta:
        :param D:
        :param vocab_size:
        :param cluster_doc_count:
        :param cluster_word_count:
        :param cluster_word_distribution:
        :return:
        """
        mgp = MovieGroupProcess(K, alpha, beta, n_iters=30)
        mgp.number_docs = D
        mgp.vocab_size = vocab_size
        mgp.cluster_doc_count = cluster_doc_count
        mgp.cluster_word_count = cluster_word_count
        mgp.cluster_word_distribution = cluster_word_distribution
        return mgp

    @staticmethod
    def _sample(p):
        """
        Sample with probability vector p from a multinomial distribution
        :param p: list
            List of probabilities representing probability vector for the multinomial distribution
        :return: int
            index of randomly selected output
        """
        return [i for i, entry in enumerate(multinomial(1, p)) if entry != 0][0]

    def _convert_doc_row(self, doc_row):
        """
        Given a vectorized doc, with each column representing
        the count of a word, return a list of words with count
        >= 1.
        """
        doc = []
        # given a row of doc, with each
        # column indicating the count of the
        # word in the doc, get the index of words
        # with counts != 0.
        for idx in np.nonzero(doc_row)[0]:
            for j in range(doc_row[idx]):
                doc.append(self.vocab[idx])
        return doc

    def fit(self, docs, DEBUG=False):
        """
        Cluster the input documents
        :param docs: list of list
            list of lists containing the unique token set of each document
        :param V: total vocabulary size for each document
        :return: list of length len(doc)
            cluster label for each document
        """
        alpha, beta, K, n_iters = self.alpha, self.beta, self.K, self.n_iters

        # new portion
        flattened = [(" ").join(doc) for doc in docs]
        vec = CountVectorizer()
        self.doc_term_matrix = vec.fit_transform(flattened).toarray()
        self.vocab = vec.get_feature_names()
        self.vocab_size = len(self.vocab)
        print("Finished vectorizing.")

        D = len(docs)
        self.number_docs = D
        # can only create doc_topic matrix
        # after getting length of docs
        self.doc_topic_matrix = np.zeros((D, K))

        # unpack to easy var names
        # n_z_w is a list of dict
        # with each dict key: word, val: count
        m_z, n_z, n_z_w = (
            self.cluster_doc_count,
            self.cluster_word_count,
            self.cluster_word_distribution,
        )
        cluster_count = K
        # list of clusters/labels
        # for every doc
        d_z = [None for i in range(len(docs))]

        vocab_freq = list(self.doc_term_matrix.sum(axis=0))
        assert len(vocab_freq) == len(self.vocab)

        # new implemetation
        # Initialize the clusters.
        for i, doc_row in enumerate(self.doc_term_matrix):
            doc = self._convert_doc_row(doc_row)

            # choose a random  initial cluster for the doc
            z = self._sample([1.0 / K for _ in range(K)])
            d_z[i] = z
            m_z[z] += 1
            n_z[z] += len(doc)  # num of words in doc

            for word in doc:
                if word not in n_z_w[z]:
                    n_z_w[z][word] = 0
                n_z_w[z][word] += 1

        print("Initialized clusters.")

        # new implementation
        # main engine.
        for _iter in tqdm(range(n_iters), desc="Running GSDMM topic model."):
            total_transfers = 0
            for i, doc_row in enumerate(self.doc_term_matrix):
                doc = self._convert_doc_row(doc_row)
                # print(f'doc = {doc}')
                assert doc_row.sum() == len(doc)
                # remove the doc from it's current cluster
                z_old = d_z[i]

                m_z[z_old] -= 1
                n_z[z_old] -= len(doc)

                for word in doc:
                    n_z_w[z_old][word] -= 1

                    # compact dictionary to save space
                    if n_z_w[z_old][word] == 0:
                        del n_z_w[z_old][word]

                # draw sample from distribution to find new cluster
                p = self.score(doc)
                # print(f"p = {p}")
                self._update_doc_topic_matrix(i, p)

                z_new = self._sample(p)

                # transfer doc to the new cluster
                if z_new != z_old:
                    total_transfers += 1
                # z is the cluster
                # d is the individual doc
                # m is the doc count
                # n is the word count
                d_z[i] = z_new
                m_z[z_new] += 1
                n_z[z_new] += len(doc)

                for word in doc:
                    if word not in n_z_w[z_new]:
                        n_z_w[z_new][word] = 0
                    n_z_w[z_new][word] += 1

            cluster_count_new = sum([1 for v in m_z if v > 0])
            if DEBUG:
                tqdm.write(
                    "In stage %d: transferred %d clusters with %d clusters populated"
                    % (_iter, total_transfers, cluster_count_new)
                )
            if (
                total_transfers == 0
                and cluster_count_new == cluster_count
                and _iter > 25
            ):
                print("Converged.  Breaking out.")
                break
            cluster_count = cluster_count_new
        # for i, topic_dict in enumerate(n_z_w):
        #     print(f"Topic {i} = {topic_dict}")
        #     print("\n")
        #     print("\n")
        # return
        # need to handle empty clusters
        # because some clusters will have 0 docs.
        idx_to_del = []
        for idx, (topic_dict, v) in enumerate(zip(n_z_w, m_z)):
            try:
                assert (v <= 0) == (not topic_dict)
            except:
                print(f"v = {v}")
                print("However, topic_dict for {idx} does not exist.")
            # if topic_dict is empty
            if not topic_dict:
                #                 print(f'cluster number {idx} is empty')
                #                 print(f'Dict for cluster {idx}')
                #                 print(n_z_w[idx])
                #                 print(f'Cluster {idx} for all docs = ')
                #                 print(self.doc_topic_matrix[:,idx])
                #                 print(f'Max prob for all the docs for cluster {idx} is {self.doc_topic_matrix[:,idx].max()}')
                # save the indices to delete later.
                # Cannot delete immediately or the index will
                # get messed up with the topic index from n_z_w and m_z.
                idx_to_del.append(idx)

        # key = old topic num
        # val = new topic num
        topic_map = {}
        new_topic = 0
        for idx, topic_dict in enumerate(n_z_w):
            # if topic_dict is not empty
            if topic_dict:
                topic_map[idx] = new_topic
                new_topic += 1
            else:
                topic_map[idx] = np.nan
        self.topic_map = topic_map
        # print(f"topic map = {topic_map}")
        assert len(n_z_w) == len(m_z) == len(n_z)
        new_m_z = []
        new_n_z = []
        new_n_z_w = []
        # the index will change here. Index num will
        # not correspond to the correct topic num.
        for idx, (m_z_item, n_z_item, n_z_w_item) in enumerate(zip(m_z, n_z, n_z_w)):
            if idx not in idx_to_del:
                new_m_z.append(m_z_item)
                new_n_z.append(n_z_item)
                new_n_z_w.append(n_z_w_item)

        # for numpy array we can delete simultaneously
        self.doc_topic_matrix = np.delete(self.doc_topic_matrix, idx_to_del, 1)

        # update internal variables
        # to number of new clusters
        self.cluster_doc_count = new_m_z
        self.cluster_word_count = new_n_z
        self.cluster_word_distribution = new_n_z_w
        self.K = cluster_count
        # print(f"New length of n_z_w = {len(self.cluster_word_distribution)}")
        # print(f"New length of m_z = {len(self.cluster_doc_count)}")
        # print(f"New length of n_z = {len(self.cluster_word_count)}")
        # print(f"New shape for doc_topic_matrix = {self.doc_topic_matrix.shape}")
        assert (
            len(self.cluster_doc_count)
            == len(self.cluster_word_count)
            == len(self.cluster_word_distribution)
            == self.K
        )

        # normalize the probabilities
        # with the new num of clusters
        self._normalize_doc_topic_matrix()

        # THIS PART IS WRONG. BECAUSE IT DID
        # NOT GO THROUGH THE SAMPLING PROCESS.
        # Even if we fix the topic renumbering,
        # e.g. if Topic 0's new topic number is actually topic 1,
        # if we use this argmax method instead of the earlier
        # sample on the final prob vector, we might end up with different
        # topics anyway. Hence topic 0 (old) might actually be topic 3 now,
        # not because of remapping of topic numbers, but literally assigning
        # into a new topic number.
        try:
            assert len(set(d_z)) == self.K
            # print(f"Unique old d_z topics = {set(d_z)}")
        except:
            print("d_z topic numbers inconsistent with K.")
            print(f"d_z has topics = {set(d_z)}")
            print(f"K has topics = {self.K}")
        d_z = self._update_d_z(d_z)
        # for old, new_ in zip(d_z, new_d_z):
        #     if old != new_:
        #         print(f"Old d_z = {old}")
        #         print(f"New d_z = {new_}")
        # assert d_z == new_d_z
        # d_z = new_d_z

        self.topic_term_list = self._topic_term()
        self._topic_term_matrix()
        try:
            assert self.topic_term_matrix.shape[0] == self.doc_topic_matrix.shape[1]
        except:
            print("Error in fit function.")
            print(f"Shape of topic_term_matrix = {self.topic_term_matrix.shape}")
            print(f"Shape of doc_topic_matrix = {self.doc_topic_matrix.shape}")
        return d_z

    def score(self, doc):
        """
        Score a document

        Implements formula (3) of Yin and Wang 2014.
        http://dbgroup.cs.tsinghua.edu.cn/wangjy/papers/KDD14-GSDMM.pdf

        :param doc: list[str]: The doc token stream
        :return: list[float]: A length K probability vector where each component represents
                              the probability of the document appearing in a particular cluster
        """
        # print("Scoring documents...")
        alpha, beta, K, V, D = (
            self.alpha,
            self.beta,
            self.K,
            self.vocab_size,
            self.number_docs,
        )

        m_z, n_z, n_z_w = (
            self.cluster_doc_count,
            self.cluster_word_count,
            self.cluster_word_distribution,
        )

        p = [0 for _ in range(K)]

        #  We break the formula into the following pieces
        #  p = N1*N2/(D1*D2) = exp(lN1 - lD1 + lN2 - lD2)
        #  lN1 = log(m_z[z] + alpha)
        #  lN2 = log(D - 1 + K*alpha)
        #  lN2 = log(product(n_z_w[w] + beta)) = sum(log(n_z_w[w] + beta))
        #  lD2 = log(product(n_z[d] + V*beta + i -1)) = sum(log(n_z[d] + V*beta + i -1))

        lD1 = log(D - 1 + K * alpha)
        doc_size = len(doc)
        for label in range(K):
            lN1 = log(m_z[label] + alpha)
            lN2 = 0
            lD2 = 0
            for word in doc:
                lN2 += log(n_z_w[label].get(word, 0) + beta)
            # for every doc
            for j in range(1, doc_size + 1):
                lD2 += log(n_z[label] + V * beta + j - 1)
                # print(f"lNl = {lN1}")
                # print(f"lD1 = {lD1}")
                # print(f"lN2 = {lN2}")
                # print(f"lD2 = {lD2}")
            # when exp(lN1 - lD1 + lN2 - lD2) is too large
            # it becomes 0. Because exp of a large neg number
            # approaches 0.
            # lN2 is a large neg num while lD2 is a large pos num.
            # Hence when lN2 - lD2, we get a large neg num.
            # if exp(lN1 - lD1 + lN2 - lD2) > 0:
            p[label] = exp(lN1 - lD1 + lN2 - lD2)
            # else:
            #     print(f"Value is {exp(lN1 - lD1 + lN2 - lD2)}")
            #     p[label] = 1e-30
            #             print(lN1 - lD1 + lN2 - lD2)
            # p[label] = exp(lN1 - lD1 + lN2 - lD2)
        # assert np.array(p).sum() != 0
        #         try:
        #             assert(np.array(p).sum() != 0)
        #         except:
        #             print(p)

        # normalize the probability vector
        pnorm = sum(p)
        pnorm = pnorm if pnorm > 0 else 1
        # try:
        #     assert np.array(p).sum() != 0
        # except:
        #     print(p)
        return [pp / pnorm if pp != 0 else 1e-30 for pp in p]

    def choose_best_label(self, doc):
        """
        Choose the highest probability label for the input document
        :param doc: list[str]: The doc token stream
        :return:
        """
        p = self.score(doc)
        return argmax(p), max(p)

    # nzw is the num of occurrences of
    # word w in cluster z. We need to
    # loop over all the words in a cluster z,
    # sum all the counts of the words in cluster z,
    def _topic_term(self):
        """
        Calculate a list of topic dictionaries,
        with each dictionary containing the probability  
        of each word occurring in its respective topic.
        """
        n_z_w = self.cluster_word_distribution
        topic_term_list = copy.deepcopy(n_z_w)
        n_z = self.cluster_word_count
        for topic_dict, topic_term_dict, total_count in zip(
            n_z_w, topic_term_list, n_z
        ):
            for word, count in topic_dict.items():
                # posterior distribution does not sum to 1
                # across all words in a cluster.
                topic_term_dict[word] = (count + self.beta) / (
                    total_count + self.vocab_size * self.beta
                )
        return topic_term_list

    def _topic_term_matrix(self):
        # create topic_term_matrix and
        # topic_term_freq_matrix with
        # new number of clusters.
        # Need to make sure that K here
        # has been updated to the new number of clusters
        # after the removal of the empty clusters.
        self.topic_term_matrix = np.zeros((self.K, len(self.vocab)))
        assert self.K == len(self.topic_term_list)
        for row, topic_dict in zip(self.topic_term_matrix, self.topic_term_list):
            word_indx = 0
            for word, prob in topic_dict.items():
                word_idx = self.vocab.index(word)
                row[word_idx] = prob

        # need to re-normalize the topic term matrix
        # because the sum of the probabilites of words
        # for each topic is not 1.
        self.topic_term_matrix = (
            self.topic_term_matrix / self.topic_term_matrix.sum(axis=1)[:, None]
        )
        try:
            assert self.topic_term_matrix.sum(axis=1).sum() == self.K
        except:
            for idx, row in enumerate(self.topic_term_matrix):
                if row.sum() != self.K:
                    print(idx)
                    print(row)
        # print(f'Shape of topic_term_matrix in _topic_term_matrix function is {self.topic_term_matrix.shape}')

    def _update_doc_topic_matrix(self, doc_num, topic_prob):
        topic_prob_array = np.array(topic_prob)
        try:
            assert topic_prob_array.sum() != 0
        except:
            print(doc_num)
        # assert(topic_prob_array.sum()==1)

        self.doc_topic_matrix[doc_num] = topic_prob
        # self.doc_topic_matrix[doc_num] = self.doc_topic_matrix[doc_num]/self.doc_topic_matrix[doc_num].sum()
        # print(doc_num)
        # print(self.doc_topic_matrix[doc_num].sum())
        # assert(self.doc_topic_matrix[doc_num].sum()== 1)

    def _normalize_doc_topic_matrix(self):
        # pass
        # np.exp(l - scipy.misc.logsumexp(l))
        self.doc_topic_matrix = (
            self.doc_topic_matrix / self.doc_topic_matrix.sum(axis=1)[:, None]
        )
        assert self.doc_topic_matrix.sum(axis=1).sum() == self.number_docs

    def _update_d_z(self, d_z):
        new_d_z = []
        for doc_cluster in d_z:
            new_d_z.append(self.topic_map[doc_cluster])
        return new_d_z
