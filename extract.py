from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np


embedder = SentenceTransformer("all-MiniLM-L6-v2")

#example
document = """
Interview started with my brief introduction including Projects and Competitive Programming profiles. Many candidates copied in the OA, so he asked me to explain all OA questions and approaches orally. I explained my approach on pen and paper, also explained how I reached that solution. He was pretty much convinced that at least I didn't copy or memorized solutions.
Then he gave me a code snippet in C language and made me identify the errors without running it. It had some syntactical errors and logical ones (like missing base case for Recursive function).
Then he gave me a simple DSA question on linked list: Remove Duplicates from Sorted List.
I started with brute force using extra space. Then we discussed the Time and Space Complexity.
He asked me to optimise it. So I explained to him a two pointer approach of Amortized Time Complexity as O(N) and O(1) Space Complexity.
He said to implement it. So I had to implement the entire question including building the linked list itself from input, printing the linked list and the solution.
It passed all test cases in one go without any error. So we moved on to the next question: Given a Target and array of integers find first and last position of Target in the array.
Again I started with brute force, implemented it (again passed all test cases in one go). We discussed its complexity. Then he just orally asked how could you optimize it. So I explained to him Upper and Lower Bound Concept.
Interview ended with question from my side.
"""


splitter = RecursiveCharacterTextSplitter(chunk_size=60, chunk_overlap=10)
chunks = splitter.split_text(document)
print("Chunks:", chunks)


embeddings = embedder.encode(chunks, convert_to_numpy=True)
print("Vector shape:", embeddings.shape)


dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)


def retrieve(query, k=2):
    query_vector = embedder.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_vector, k)
    results = [chunks[i] for i in indices[0]]
    return results, distances


query = "In which language the code snippet was given?"
results, distances = retrieve(query)
print("\nQuery:", query)
print("Retrieved Chunks:", results)
