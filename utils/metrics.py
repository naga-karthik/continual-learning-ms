"""
Implements the CL metrics defined in GEM (https://arxiv.org/pdf/1706.08840.pdf) 
and TAG (https://arxiv.org/pdf/2105.05155.pdf) papers
"""

import numpy as np


def BWT(result_matrix):
        """
        Backward Transfer metric

        :param result_matrix: TxT matrix containing accuracy values in each (i, j) entry.
        (i, j) -> test accuracy on task j after training on task i
        """

        final_accs = result_matrix[-1, :]  # take accuracies after final training
        # accuracies on task i right after training on task i, for all i
        training_accs = np.diag(result_matrix)
        task_bwt = final_accs - training_accs  # BWT for each task
        average_bwt = np.mean(task_bwt)  # compute average
        return average_bwt, task_bwt

def FWT(result_matrix, single_task_res):
        """
        Forward Transfer metric
        :param result_matrix: TxT matrix containing accuracy values in each (i, j) entry.
                (i, j) -> test accuracy on task j after training on task i
        :param single_task_res: 1xT matrix containing single-task accuracies on random init
        """
        num_domains = result_matrix.shape[0]
        task_fwt = np.zeros(num_domains)
        for k in range(1, num_domains):
                task_fwt[k] = result_matrix[k - 1, k] - single_task_res[k]
        avg_fwt = np.mean(task_fwt)
        return avg_fwt, task_fwt


def ACC(result_matrix):
        """
        Average Accuracy metric

        :param result_matrix: TxT matrix containing accuracy values in each (i, j) entry.
        (i, j) -> test accuracy on task j after training on task i
        """
        final_accs = result_matrix[-1, :]  # take accuracies after final training
        acc = np.mean(final_accs)  # compute average
        return acc, final_accs

def LA(result_matrix):
        """
        Learning Accuracy metric

        :param result_matrix: TxT matrix containing accuracy values in each (i, j) entry.
        (i, j) -> test accuracy on task j after training on task i
        returns the test accuracy on a task immediately after training on that task
        """ 
        
        learning_accs = np.diag(result_matrix)

        return learning_accs
