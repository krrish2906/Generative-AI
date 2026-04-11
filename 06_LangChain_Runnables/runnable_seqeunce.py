from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence
import os

load_dotenv()

model = ChatOpenAI(
    model="meta-llama/llama-3-8b-instruct",
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1"
)
parser = StrOutputParser()

prompt1 = PromptTemplate(
    template='Write a joke about {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Explain the following joke - {text}',
    input_variables=['text']
)

chain = RunnableSequence(prompt1, model, parser, prompt2, model, parser)
print(chain.invoke({'topic':'engineers'}))

# Output:-
"""
This joke is a play on the classic joke "Why did the chicken cross the road?" which typically has a punchline like "To get to the other side!"

The original joke is a classic example of a "dumb joke" – it's simple, repetitive, and the punchline is obvious. The humor comes from the unexpected twist on the typical format of a joke, and the silly situation being described.

In this variation, the punchline is "that they designed" – implying that the engineer not only crossed the road, but also designed the road itself. It's a clever play on the idea of engineers being responsible for designing and building infrastructure like roads, and it adds a layer of complexity and cleverness to the joke.

The humor comes from the unexpected twist on the typical format, as well as the dry, witty delivery – it's not necessarily laugh-out-loud hilarious, but it might elicit a chuckle or a nod of appreciation for the clever wordplay.
"""