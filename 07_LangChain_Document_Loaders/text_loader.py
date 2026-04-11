from langchain_community.document_loaders import TextLoader
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
    template='Summarize the following text: \n{text}',
    input_variables=['text']
)

parser = StrOutputParser()
loader = TextLoader('data.txt', encoding='utf-8')

docs = loader.load()

print("Type of docs:", type(docs), "\n")
print("Number of documents:", len(docs), "\n")

chain = prompt | model | parser

response = chain.invoke({'text': docs[0].page_content})
print("Response:", response, "\n")

# Output:-
"""
Type of docs: <class 'list'> 

Number of documents: 1 

Response: The text discusses the intersection of Artificial Intelligence (AI) and society, covering its fundamentals, applications, and implications. Key points include:

**Key Components of AI:**

1. **Machine Learning**: A subset of AI that enables models to learn from data and improve over time.
2. **Data**: High-quality data is crucial for accurate model performance, and preprocessing steps like cleaning and feature engineering are essential.
3. **Natural Language Processing (NLP)**: Allows machines to understand and generate human language.
4. **Computer Vision**: Enables machines to interpret and understand visual data like images and videos.

**Challenges and Considerations:**

1. **Ethical Concerns**: Issues like bias, lack of transparency, and potential misuse of AI arise as AI systems become more powerful.
2. **Responsibility**: Fairness, accountability, and transparency are essential in AI development.

**Future of AI:**

1. **Advancements**: Generative AI, robotics, and human-computer interaction will shape the next generation of AI applications.
2. **Regulation and Governance**: Crucial for ensuring AI benefits society as a whole.

**Key Takeaways:**

1. AI is transforming daily life through virtual assistants, autonomous vehicles, and more.
2. Understanding AI fundamentals, challenges, and implications is crucial in this rapidly evolving field.
3. Continuous learning and adaptation will be essential for staying relevant in the AI landscape.
"""
