import os
from extract_text import extract_text_from_folder
from feature_extraction import extract_features, vectorize_texts
from similarity import compute_cosine_similarity, compute_jaccard_similarity, combine_similarities

def match_invoices(test_folder, train_folder):
    train_texts = extract_text_from_folder(train_folder)
    test_texts = extract_text_from_folder(test_folder)
    
    train_features = {filename: extract_features(text) for filename, text in train_texts.items()}
    test_features = {filename: extract_features(text) for filename, text in test_texts.items()}
    
    train_vector_texts = [features['text'] for features in train_features.values()]
    test_vector_texts = [features['text'] for features in test_features.values()]

    train_vectors, vectorizer = vectorize_texts(train_vector_texts)
    test_vectors = vectorizer.transform(test_vector_texts)
    
    results = {}
    
    for test_filename, test_feature in test_features.items():
        max_similarity = 0
        best_match = None
        
        test_vector = test_vectors[list(test_features.keys()).index(test_filename)]
        
        for train_filename, train_feature in train_features.items():
            train_vector = train_vectors[list(train_features.keys()).index(train_filename)]
            
            cosine_sim = compute_cosine_similarity(test_vector, train_vector)
            jaccard_sim = compute_jaccard_similarity(test_feature['keywords'], train_feature['keywords'])
            
            combined_similarity = combine_similarities(cosine_sim, jaccard_sim)
            
            if combined_similarity > max_similarity:
                max_similarity = combined_similarity
                best_match = train_filename
        
        results[test_filename] = (best_match, max_similarity)
    
    return results

if __name__ == "__main__":
    test_folder = "data/test"
    train_folder = "data/train"
    results = match_invoices(test_folder, train_folder)
    for test_file, (train_file, score) in results.items():
        print(f"Test Invoice: {test_file} -> Most Similar Train Invoice: {train_file} with similarity score of {score:.2f}")
