from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough
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

joke_generator_chain = RunnableSequence(prompt1, model, parser)

parallel_chain = RunnableParallel({
    'joke': RunnablePassthrough(),
    'explanation': RunnableSequence(prompt2, model, parser)
})

final_chain = RunnableSequence(joke_generator_chain, parallel_chain)
response = final_chain.invoke({'topic':'engineers'})

print("Joke:", response['joke'], end='\n\n')
print("Explanation:", response['explanation'])

# Output:-
"""
Joke: Why did the engineer cross the road?

Because he was trying to optimize the route and minimize the coefficient of friction, while also ensuring the structural integrity of the pavement and adhering to all relevant safety protocols... but ultimately, he still got stuck in traffic!

Explanation: Ahaha! This joke is a clever play on the classic "why did the chicken cross the road" joke, but with a twist. Here's a breakdown:

**Traditional setup:** The classic joke typically goes like this: "Why did the X (chicken, cow, etc.) cross the road?" followed by a punchline that's often a joke about the reasons for crossing (e.g., "To get to the other side!").

**Keyword "Optimize":** In software development, finance, and other fields, an "optimizer" is someone who seeks to optimize a process or a function to get the best possible outcome. Engineers, in particular, might want to optimize routes to reduce time, energy, or cost. So, the joke starts by setting up the expectation that the engineer is trying to "optimize" their route.

**Minimize the coefficient of friction:** This engineering term refers to the idea of reducing the resistance or friction that occurs when an object moves over a surface. In the context of the joke, this means the engineer wants to minimize the friction to make the crossing more efficient. The joke is now acknowledging that the engineer is getting technical and considering some of the more tedious aspects of transportation engineering.

**Ensure structural integrity of the pavement:** The engineer wants to ensure the pavement won't collapse or break under the weight of the road vehicles. This phrase raises the stakes and makes the joke sound like it's poking fun at the very serious nature of engineering.

**Adhering to all relevant safety protocols:** This adds another layer of engineering nicety: the engineer wants to follow all the regulations and take care to avoid accidents.

**But ultimately, the engineer gets stuck in traffic:** Ah, the punchline! This is where the joke has a hilarious little twist. The engineer's careful planning and involvement in the fields of engineering don't make them immune to getting stuck in traffic. It's almost like they're as unaware of the chaos of traffic as non-engineers are.

The humor in this joke lies in the lighthearted play with the engineering terminology, self-serious employment of technical jargon, and the relatable yet also parodic portrayal of an engineer being stuck in traffic despite all their efforts to "optimize" and "optimize" and "improve."
"""
