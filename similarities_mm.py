import numpy as np
import math
# Similarities functions

def jaccard_similarity(x ,y):
    intersection_cardinality = len(np.intersect1d(x,y))
    union_cardinality = len(np.union1d(x, y))
    #intersection_cardinality = len(set.intersection(*[set(x), set(y)]))
    #union_cardinality = len(set.union(*[set(x), set(y)]))
    return intersection_cardinality /float(union_cardinality)

def square_rooted(x):
    return round(math.sqrt(sum([ a *a for a in x])) ,3)

def cosine_similarity(x ,y):
    numerator = sum( a *b for a ,b in zip(x ,y))
    denominator = square_rooted(x )*square_rooted(y)
    return round(numerator /float(denominator) ,5)