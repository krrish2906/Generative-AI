from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Chat Template with Message Placeholder
chat_template = ChatPromptTemplate([
    ('system', 'You are a helpful customer support agent.'),
    MessagesPlaceholder(variable_name='chat_history'),
    ('human', '{query}')
])

# Load Chat history
chat_history = []
with open('chat_history.txt', 'r') as f:
    chat_history.extend(f.readlines())

print(chat_history)

# Create a Prompt
prompt = chat_template.invoke({
    'chat_history': chat_history,
    'query': 'What is the status of my refund?'
})
print(prompt)