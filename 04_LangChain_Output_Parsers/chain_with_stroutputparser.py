from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

template1 = PromptTemplate(
    template="Write a detailed report on {topic}",
    input_variables=["topic"]
)

template2 = PromptTemplate(
    template="Write a 5 line summary on the following text:\n {text}",
    input_variables=["text"]
)

parser = StrOutputParser()
chain = template1 | model | parser | template2 | model | parser

response = chain.invoke({"topic": "Artificial Intelligence"})
print(response)

# Output:-
"""
Here's a 5 line summary of the Artificial Intelligence (AI) Report:

Artificial Intelligence (AI) is a rapidly growing field that enables machines to perform tasks requiring human intelligence. AI has various applications in fields such as healthcare, finance, transportation, and education, and its benefits include improved efficiency, enhanced decision-making, and increased accuracy. AI types include Narrow/Weak, General/Strong, and Superintelligence, with Narrow AI being the most common. The applications of AI include medical diagnosis, financial analysis, self-driving cars, and personalized education. Overall, AI has the potential to transform industries and improve lives, but also poses significant challenges and risks.
"""