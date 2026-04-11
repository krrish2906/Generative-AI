from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

model = ChatOpenAI(
    model="meta-llama/llama-3-8b-instruct",
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1"
)

prompt = PromptTemplate(
    template='Answer the following question: \n{question}\nfrom the following text: \n{text}',
    input_variables=['question','text']
)

parser = StrOutputParser()
url = 'https://www.amazon.in/Alisha-Craft-Furniture-Cushion-Protected/dp/B0F6MB6TQ7/?_encoding=UTF8&pd_rd_w=eA1jK&content-id=amzn1.sym.eab426f1-2a63-454c-85b6-64d91fd491ec&pf_rd_p=eab426f1-2a63-454c-85b6-64d91fd491ec&pf_rd_r=7ZNQ3YNHP2P3GR5F8V4V&pd_rd_wg=F2nN4&pd_rd_r=63394c34-2d67-4913-8102-75300fa0da75&th=1'

loader = WebBaseLoader(url)
docs = loader.load()

chain = prompt | model | parser
text = docs[0].page_content
questions = [
    "What is the product used for?",
    "What are the key features of the product?",
    "What is the price of the product?"
]

for question in questions:
    response = chain.invoke({'question': question, 'text': text})
    print(f"Question: {question}")
    print(f"Answer: {response}")
    print()


# Output:-
"""
Question: What is the product used for?
Answer: According to the text, the product is a single seater swing chair, and it is used for outdoor/indoor decoration, relaxation, or leisure activities. It can be placed in a backyard patio, deck, in a sunroom or garden, or near a pool, outdoor bar, or living room.

Question: What are the key features of the product?
Answer: According to the text, the key features of the product are:

1. Ideal for Deck, balcony & more - Unique swinging Chair is the perfect addition to any space outside, like a backyard patio, deck, in a sunroom or garden, or near a pool, or outdoor bar.
2. Premium Fluffy Cushion - It comes with a Soft deep fluffy cushion, perfect for you to snuggle in on a warm and sunny day, with a book and a cold coffee
3. User Indoor or Outdoors - It is sturdy & safe for you to sit in, and it will be a great addition to your indoor or outdoor furniture.
4. Powder Coated Frame, UV Protected Wicker.
5. The cushion material is 100% polyester.
6. Recommended load: Max. 125kg.
7. The product is water-resistant.
8. The hook is attached to a stable fixture on the ceiling.
9. The product can easily be removed and relocated from indoor to outdoor.

Question: What is the price of the product?
Answer: According to the text, the price of the product is ₹9,895.00.
"""