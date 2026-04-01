from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)

class Person(BaseModel):
    name: str = Field(description='Name of the person')
    age: int = Field(gt=18, description='Age of the person')
    city: str = Field(description='Name of the city the person belongs to')

model = ChatHuggingFace(llm=llm)
parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template="""
    Generate a fictional {place} person.
    Return ONLY a valid JSON object.
    Do NOT include:
        - explanations
        - code
        - markdown
        - backticks

    {format_instruction}
    """,
    input_variables=["place"],
    partial_variables={"format_instruction": parser.get_format_instructions()}
)

# =========================================================================

# prompt = template.format(place='Indian')
# print("Prompt:", prompt, end='\n\n')

# response = model.invoke(prompt)
# print("Response:", response, end='\n\n')

# parsed_response = parser.parse(response.content)
# print("Parsed Response:", parsed_response)

# =========================================================================
# Output:-
"""
Prompt: 
    Generate a fictional Indian person.
    Return ONLY a valid JSON object.
    Do NOT include:
        - explanations
        - code
        - markdown
        - backticks

    The output should be formatted as a JSON instance that conforms to the JSON schema below.

As an example, for the schema {"properties": {"foo": {"title": "Foo", "description": "a list of strings", "type": "array", "items": {"type": "string"}}}, "required": ["foo"]}
the object {"foo": ["bar", "baz"]} is a well-formatted instance of the schema. The object {"properties": {"foo": ["bar", "baz"]}} is not well-formatted.

Here is the output schema:
```
{"properties": {"name": {"description": "Name of the person", "title": "Name", "type": "string"}, "age": {"description": "Age of the person", "exclusiveMinimum": 18, "title": "Age", "type": "integer"}, "city": {"description": "Name of the city the person belongs to", "title": "City", "type": "string"}}, "required": ["name", "age", "city"]}
```

Response: content='{"name": "Raghav Raj", "age": 32, "city": "Mumbai"}' additional_kwargs={} response_metadata={'token_usage': {'completion_tokens': 23, 'prompt_tokens': 289, 'total_tokens': 312}, 'model_name': 'meta-llama/Llama-3.1-8B-Instruct', 'system_fingerprint': 'fp_ff5ddd64106da7b20c62', 'finish_reason': 'stop', 'logprobs': None} id='lc_run--019d498a-d80c-7c01-961e-5b2623533939-0' tool_calls=[] invalid_tool_calls=[] usage_metadata={'input_tokens': 289, 'output_tokens': 23, 'total_tokens': 312}

Parsed Response: name='Raghav Raj' age=32 city='Mumbai'
"""

# =========================================================================

chain = template | model | parser
response = chain.invoke({"place": "Indian"})
print("Response:", response, end="\n\n")
print("Type:", type(response))

# =========================================================================
# Output:-
"""
Response: name='Rajeshwar Kumar' age=35 city='Mumbai'

Type: <class '__main__.Person'>
"""
# =========================================================================