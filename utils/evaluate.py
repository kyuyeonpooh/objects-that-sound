import math
import os

import numpy as np

from ontology import Ontology


def get_max_tree_distance(tags):
    """
    Description:
        Return max tree distance which can be derived with given tag list.
    Parameters:
        tags: list of tags used in training. (type: list)
              [Example] ['Acoustic guitar', 'Electric Guitar', ..., 'Piano']
    """
    return max_dist


def dist_to_score(distances):
    """
    Description:
        Convert distances of K retrieved items into scores
    Parameters:
        distances: tree distance between query and K retrieved items
                   [Example] [0, 0, 1, 2, 1, 0, 5, 4, ..., 9] (type: ndarray, len: K)
    [Note] score = max_tree_distance - distance
    """
    return scores


def DCG(scores):
    """
    Description:
        Return DCG(Discounted Cumulative Gain) with given score (relevance) list
    Parameters:
        scores: score list of K retrieved items (type: ndarray, len: K)
              [Example] [8, 6, 6, 8, 4, 7, ..., 2]         
    """
    return dcg


def IDCG(scores):
    """
    Description:
        Return IDCG(Ideal Discounted Cumulative Gain) with given score (relevance) list
    Parameters:
        scores: score list of K retrieved items (type: ndarray, len: K)
              [Example] [8, 6, 6, 8, 4, 7, ..., 2]  
    """
    return idcg


def nDCG(scores):
    """
    Description:
        Return nDCG(normalized Discounted Cumulative Gain) with given score (relevance) list
    Parameters:
        scores: score list of K retrieved items (type: ndarray, len: K)
              [Example] [8, 6, 6, 8, 4, 7, ..., 2]
                
    """
    return ndcg
