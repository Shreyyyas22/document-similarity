# Document Similarity Project

## Approach

### Document Representation Method
I have chosen TF-IDF (Term Frequency-Inverse Document Frequency) vectorization as our document representation method. This technique transforms the text documents into numerical vectors, capturing the importance of each word in a document relative to the entire corpus. By doing so, it highlights significant words while downplaying common ones, thus providing a more meaningful representation of the document's content.

### Similarity Metrics Used
To measure the similarity between documents, we utilize two metrics:
1. **Cosine Similarity**: This metric evaluates the cosine of the angle between two vectors, providing a measure of how similar the documents are in terms of their content. It ranges from 0 to 1, where 1 indicates that the documents are identical in terms of content direction.
2. **Jaccard Similarity**: This metric measures the overlap between the sets of keywords extracted from the documents. It is defined as the size of the intersection divided by the size of the union of the keyword sets, ranging from 0 to 1. Higher values indicate more significant keyword overlap.

I combine these two similarity scores using a weighted average to obtain a final similarity score, which balances both content direction and keyword overlap.

## Instructions

### Requirements
- Python 3.6 or higher
- `re`
- `string`
- `pdfplumber`
- `nltk`
- `scikit-learn`

 
- Ensure that the NLTK stopwords corpus is downloaded:
import nltk
nltk.download('stopwords')

## Running the Code
Place the test PDF files in the data/test directory.
Place the training PDF files in the data/train directory.
Run the main.py script


