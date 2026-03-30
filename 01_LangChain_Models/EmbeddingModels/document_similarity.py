from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from dotenv import load_dotenv

load_dotenv()

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
documents = [
    "Sachin Tendulkar, also known as the 'God of Cricket', holds many batting records.",
    "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills.",
    "Rohit Sharma is known for his elegant batting and record-breaking double centuries.",
    "Virat Kohli is an Indian cricketer known for his aggressive batting and leadership.",
    "Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers."
]

# query = "Tell me about the god of cricket"
query = "Tell me about Virat Kohli"

doc_embeddings = embeddings.embed_documents(documents)
query_embedding = embeddings.embed_query(query)

similarities = cosine_similarity([query_embedding], doc_embeddings)[0]
similarities = sorted(list(enumerate(similarities)), key=lambda x: x[1], reverse=True)

for s in similarities:
    print(f"Document {s[0]+1}: {s[1]}")

index, similarity_score = similarities[0]
print("-" * 100)
print(f"Most similar document is document {index + 1} with similarity: {similarity_score}")

print("-" * 100)
print("User: " + query)
print("Assistant: " + documents[index])
print("Similarity: " + str(similarity_score))
print("-" * 100)