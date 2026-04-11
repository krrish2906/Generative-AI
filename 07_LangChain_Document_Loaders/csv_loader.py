from langchain_community.document_loaders import CSVLoader
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
loader = CSVLoader(file_path='Student_data.csv')
docs = loader.load()

chain = prompt | model | parser
text = "\n".join([doc.page_content for doc in docs])
questions = [
    "How many unique departments are there in total?",
    "Who is the topper of the class based on highest CGPA?",
    "How many students are there in total who have done internship?"
]

for i, question in enumerate(questions, 1):
    response = chain.invoke({'question': question, 'text': text})
    print(f"Question {i}: {question}")
    print(f"Answer: {response}")
    print()


# Output:-
"""
Question 1: How many unique departments are there in total?
Answer: There are 5 unique departments mentioned in the text:

1. CE (Computer Engineering)
2. IT (Information Technology)
3. ENTC (Electronics and Telecommunication Engineering)
4. ECE (Electrical and Computer Engineering)
5. AIDS (Artificial Intelligence and Data Science)

Question 2: Who is the topper of the class based on highest CGPA?
Answer: Based on the highest CGPA, the topper of the class is Vihaan Gupta with a CGPA of 9.8.

Question 3: How many students are there in total who have done internship?
Answer: From the text, we can see that the following students have done an internship:

1. Aarav Sharma (roll_no: 1)
2. Ishita Patel (roll_no: 2)
3. Vihaan Gupta (roll_no: 4)
4. Emily Brown (roll_no: 7)
5. Yuki Tanaka (roll_no: 11)
6. Aditya Joshi (roll_no: 16)
7. Govinda Ahuja (roll_no: 86)
8. Karisma Kapoor (roll_no: 87)
9. Kareena Kapoor (roll_no: 88)
10. Preity Zinta (roll_no: 91)
11. Juhi Chawla (roll_no: 92)
12. Neetu Singh (roll_no: 94)
13. Sridevi Kapoor (roll_no: 95)
14. Prem Chopra (roll_no: 98)
15. Emily Brown (roll_no: 7)

There are 15 students who have done an internship.
"""
