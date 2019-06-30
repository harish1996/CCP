from sklearn.tree._criterion cimport ClassificationCriterion
from sklearn.tree._criterion cimport SIZE_t

import numpy as np 

from libc.math cimport sqrt,pow,log

cdef class CCPCriterion(ClassificationCriterion):
    cdef double node_imprity(self) nogil:

        """Evaluate the impurity of the current node, i.e. the impurity of
        samples[start:end], using the cross-entropy criterion."""

        cdef SIZE_t* n_classes = self.n_classes
        cdef double* sum_total = self.sum_total
        cdef double entropy = 0.0
        cdef double count_k
        cdef SIZE_t k
        cdef SIZE_t c

        for k in range(self.n_outputs):
            for c in range(n_classes[k]):
                count_k = sum_total[c]
                if count_k > 0.0:
                    count_k /= self.weighted_n_node_samples
                    entropy -= count_k * log(count_k)

                sum_total += self.sum_stride

        return entropy / self.n_outputs


    cdef double Imp(double p, double n) nogil:
        return - ( p*log(p) ) - ( n*log(n) )

    cdef double children_impurity(self,double* impurity_left,double* impurity_right) nogil:
        """Evaluate the impurity in children nodes

        i.e. the impurity of the left child (samples[start:pos]) and the
        impurity the right child (samples[pos:end]).

        Parameters
        ----------
        impurity_left : double pointer
            The memory address to save the impurity of the left node
        impurity_right : double pointer
            The memory address to save the impurity of the right node
        """

        cdef SIZE_t* n_classes = self.n_classes
        cdef double* sum_left = self.sum_left
        cdef double* sum_right = self.sum_right
        cdef double ccp_left = 0.0,ccp_right = 0.0
        cdef double tpr=0.0,fpr=0.0

        cdef SIZE_t k,c
        
        # stop splitting in case reached pure node with 0 samples of second class
        if sum_left[1] + sum_right[1] == 0:
            impurity_left[0] = -INFINITY
            impurity_right[0] = -INFINITY
            return
        
        for k in range(self.n_outputs):
            if(sum_left[0] + sum_right[0] > 0):
                fpr = sum_left[0] / (sum_left[0] + sum_right[0])
                if(sum_left[1] + sum_right[1] > 0):
                    tpr = sum_left[1] / (sum_left[1] + sum_right[1])
        
                ccp_left -= (tpr+fpr) * Imp( tpr/(tpr+fpr), fpr/(tpr+fpr) )
                ccp_right -= ( 2- tpr-fpr ) * Imp( (1-tpr)/(2-tpr-fpr), (1-fpr)/(2-tpr-fpr) )
        
                sum_left += self.sum_stride
                sum_right += self.sum_stride
        
        impurity_left[0] = ccp_left
        impurity_right[0] = ccp_right
        
        return
        


