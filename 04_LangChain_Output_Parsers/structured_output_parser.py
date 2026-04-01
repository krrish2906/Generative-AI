from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_classic.output_parsers import StructuredOutputParser, ResponseSchema

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

schema = [
    ResponseSchema(name="fact_1", description="Fact 1 about the topic"),
    ResponseSchema(name="fact_2", description="Fact 2 about the topic"),
    ResponseSchema(name="fact_3", description="Fact 3 about the topic")
]

parser = StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(
    template="Give me 3 facts about {topic}\n{format_instructions}",
    input_variables=["topic"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

chain = template | model | parser
result = chain.invoke({"topic": "Space"})
print("Result:", result)

# Output:-
"""
Result: {'fact_1': 'The Andromeda Galaxy, our closest galactic neighbor, is approaching Earth at a speed of about 250,000 miles per hour.', 'fact_2': "The International Space Station orbits the Earth at an altitude of around 250 miles (400 kilometers) above the planet's surface.", 'fact_3': 'The farthest human-made object, Voyager 1, has traveled over 14 billion miles (22.5 billion kilometers) into interstellar space.'}
"""

# Note:
# This file demonstrates StructuredOutputParser from older LangChain versions.
# In LangChain v1.x, this approach is deprecated.
# Modern approach uses:
#   - JSON output parser
#   - Pydantic Output Parser
#
# StructuredOutputParser is now available in the `langchain-classic` package for backward compatibility.
# This file is kept for learning / reference purposes only.