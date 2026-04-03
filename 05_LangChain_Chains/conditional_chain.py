from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnableBranch, RunnableLambda
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()

class Feedback(BaseModel):
    sentiment: Literal["positive", "negative"] = Field(description="Give the sentiment of the feedback")

parser = StrOutputParser()
parser2 = PydanticOutputParser(pydantic_object=Feedback)

model = ChatGroq(model="llama-3.3-70b-versatile")

prompt1 = PromptTemplate(
    template='Classify the sentiment of the following feedback text into positive or negative: \n{feedback}\n{format_instructions}',
    input_variables=['feedback'],
    partial_variables={'format_instructions': parser2.get_format_instructions()}
)

classifier_chain = prompt1 | model | parser2

prompt2 = PromptTemplate(
    template='Write an appropriate response to this positive feedback: \n{feedback}',
    input_variables=['feedback']
)

prompt3 = PromptTemplate(
    template='Write an appropriate response to this negative feedback: \n{feedback}',
    input_variables=['feedback']
)

branch_chain = RunnableBranch(
    (lambda x: x.sentiment == "positive", prompt2 | model | parser),
    (lambda x: x.sentiment == "negative", prompt3 | model | parser),
    RunnableLambda(lambda x: "Could not classify the sentiment")
)

chain = classifier_chain | branch_chain

response1 = chain.invoke({"feedback": "This product is amazing!"})
print(response1, end="\n\n")

# Output:-
"""
Thank you so much for your kind words. I'm thrilled to hear that you're satisfied, and I appreciate the time you took to share your positive experience. If you have any other questions or need further assistance, don't hesitate to reach out.
"""

response2 = chain.invoke({"feedback": "This product is terrible!"})
print(response2, end="\n\n")

# Output:-
"""
I'm so sorry to hear that you're not satisfied. Can you please provide more details about what didn't meet your expectations? Your feedback is invaluable in helping us improve, and I'd be happy to try and make things right.
"""

chain.get_graph().print_ascii()
chain.get_graph().draw_mermaid_png(output_file_path="conditional_chain.png")

"""
    +-------------+      
    | PromptInput |      
    +-------------+      
            *
            *
            *
   +----------------+    
   | PromptTemplate |    
   +----------------+
            *
            *
            *
      +----------+
      | ChatGroq |
      +----------+
            *
            *
            *
+----------------------+
| PydanticOutputParser |
+----------------------+
            *
            *
            *
       +--------+
       | Branch |
       +--------+
            *
            *
            *
    +--------------+
    | BranchOutput |
    +--------------+
"""