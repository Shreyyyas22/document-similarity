from sklearn.metrics.pairwise import cosine_similarity

def compute_cosine_similarity(vec1, vec2):
    return cosine_similarity(vec1, vec2)[0][0]

def compute_jaccard_similarity(set1, set2):
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union

def combine_similarities(cosine_sim, jaccard_sim, weights=(0.5, 0.5)):
    return weights[0] * cosine_sim + weights[1] * jaccard_sim
