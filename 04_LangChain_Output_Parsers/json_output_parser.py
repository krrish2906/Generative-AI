from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)
parser = JsonOutputParser()

template = PromptTemplate(
    template="Give me the name, age and city of a fictional person\n{format_instructions}",
    input_variables=[],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

# =========================================================================

# prompt = template.format()
# print("Prompt:", prompt, end="\n\n")

# response = model.invoke(prompt)
# print("Response:", response, end="\n\n")

# parsed_response = parser.parse(response.content)
# print("Parsed Response:", parsed_response, end="\n\n")
# print("Type:", type(parsed_response))

# =========================================================================
# Output:-
"""
Prompt: Give me the name, age and city of a fictional person
Return a JSON object.

Response: content='```json\n{\n  "name": "Alice Thompson",\n  "age": 32,\n  "city": "New York"\n}\n```' additional_kwargs={} response_metadata={'token_usage': {'completion_tokens': 30, 'prompt_tokens': 53, 'total_tokens': 83}, 'model_name': 'meta-llama/Llama-3.1-8B-Instruct', 'system_fingerprint': 'fp_ff5ddd64106da7b20c62', 'finish_reason': 'stop', 'logprobs': None} id='lc_run--019d4929-9c86-7672-81a7-8ba04eadd32c-0' tool_calls=[] invalid_tool_calls=[] usage_metadata={'input_tokens': 53, 'output_tokens': 30, 'total_tokens': 83}

Parsed Response: {'name': 'Alice Thompson', 'age': 32, 'city': 'New York'}

Type: <class 'dict'>
"""

# =========================================================================

chain = template | model | parser
response = chain.invoke({})
print("Response:", response, end="\n\n")
print("Type:", type(response))

# =========================================================================
# Output:-
"""
Response: {'name': 'Emily Wilson', 'age': 28, 'city': 'Los Angeles'}

Type: <class 'dict'>
"""
# =========================================================================