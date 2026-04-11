from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader('AI_ML_Honor.pdf')
docs = loader.load()

print("Number of pages:", len(docs), "\n")
print("Type of docs:", type(docs), "\n")
print("Type of first document:", type(docs[0]), "\n")
print("First document:", docs[0].page_content, "\n")
print("First document metadata:", docs[0].metadata, "\n")

# Output:-
"""
Number of pages: 16 

Type of docs: <class 'list'> 

Type of first document: <class 'langchain_core.documents.base.Document'>

First document: Faculty of Science and Technology
Savitribai Phule Pune University
Maharashtra, India

http://unipune.ac.in

Honours* in Artificial Intelligence and
Machine Learning
Board of Studies
 ( Computer Engineering)
    (with effect from A.Y. 2020-21)

First document metadata: {'producer': 'Microsoft® Office Word 2007', 'creator': 'Microsoft® Office Word 2007', 'creationdate': 'D:20210330070428', 'author': 'Dr. Nuzhat F. Shaikh', 'moddate': 'D:20210330070428', 'source': 'AI_ML_Honor.pdf', 'total_pages': 16, 'page': 0, 'page_label': '1'}
"""