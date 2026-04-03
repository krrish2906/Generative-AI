from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
import os

load_dotenv()
parser = StrOutputParser()

prompt1 = PromptTemplate(
    template='Generate short and simple notes from the following text: \n{text}',
    input_variables=['text']
)

prompt2 = PromptTemplate(
    template='Generate a quiz from the following text: \n{text}',
    input_variables=['text']
)

prompt3 = PromptTemplate(
    template='Merge the following notes and quiz into a single document: \n{notes}\n\n{quiz}',
    input_variables=['notes', 'quiz']
)

model1 = ChatOpenAI(
    model="meta-llama/llama-3-8b-instruct",
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1"
)

model2 = ChatGroq(model="llama-3.3-70b-versatile")

parallel_chain = RunnableParallel({
    'notes': prompt1 | model1 | parser,
    'quiz': prompt2 | model2 | parser
})

merged_chain = prompt3 | model1 | parser
chain = parallel_chain | merged_chain

text = """Linear Regression is a fundamental supervised learning algorithm used to model the relationship between a dependent variable and one or more independent variables. It predicts continuous values by fitting a straight line that best represents the data.

Linear regression is a fundamental statistical method used in predictive analysis, serving as the bedrock for many machine learning algorithms. It models the relationship between a dependent variable and one or more independent variables, using a linear approach to establish trends in the data. The aim of this article is twofold: firstly, to delve into the theory underlying linear regression, unraveling its mathematical principles and assumptions; and secondly, to provide a hands-on demonstration of implementing linear regression in Python. By bridging theory with practice, this article aims to offer a comprehensive understanding of linear regression, making it accessible for those new to the field and a refresher for experienced practitioners.

- It assumes that there is a linear relationship between the input and output
- Uses a best‑fit line to make predictions
- Commonly used in forecasting, trend analysis, and predictive modelling 

For example we want to predict a student's exam score based on how many hours they studied. We observe that as students study more hours, their scores go up. In the example of predicting exam scores based on hours studied.
Here,
- Independent variable (input): Hours studied because it's the factor we control or observe.
- Dependent variable (output): Exam score because it depends on hobw many hours were studied.
- We use the independent variable to predict the dependent variable."""

response = chain.invoke({'text': text})
print(response)

print("\n" + "=" * 100 + "\n")
chain.get_graph().print_ascii()
chain.get_graph().draw_mermaid_png(output_file_path="parallel_chain.png")

# Output:-
"""
Here are the merged notes and quiz in a single document:

**Linear Regression**

Supervised learning algorithm that models relationship between:

* Dependent variable (continuous value)
* One or more independent variables
Predicts continuous values by fitting a straight line
Assumptions:
* Linear relationship between input and output
Uses a best-fit line to make predictions
Application:
* Forecasting
* Trend analysis
* Predictive modeling

**Example**

* Predicting exam score using hours studied as input (independent variable)
* Variables:
        + Independent (input): Hours studied
        + Dependent (output): Exam score

**Linear Regression Quiz**

**1. What is the primary purpose of Linear Regression in machine learning?**

a) To classify categorical data
b) To predict continuous values by fitting a straight line
c) To cluster similar data points
d) To reduce dimensionality

**Answer:** b) To predict continuous values by fitting a straight line

**2. What is the fundamental assumption of Linear Regression?**

a) Non-linear relationship between input and output
b) Linear relationship between input and output
c) No relationship between input and output
d) Random relationship between input and output

**Answer:** b) Linear relationship between input and output

**3. What is the role of the best-fit line in Linear Regression?**

a) To identify outliers in the data
b) To make predictions
c) To visualize the data
d) To transform the data

**Answer:** b) To make predictions

**4. What are some common applications of Linear Regression?**

a) Forecasting, trend analysis, and predictive modeling
b) Clustering, classification, and dimensionality reduction
c) Data visualization, data mining, and data warehousing
d) Machine learning, deep learning, and natural language processing

**Answer:** a) Forecasting, trend analysis, and predictive modeling

**5. In the example of predicting exam scores based on hours studied, what is the independent variable?**

a) Exam score
b) Hours studied
c) Student's age
d) Student's gender

**Answer:** b) Hours studied

**6. What is the dependent variable in the example of predicting exam scores based on hours studied?**

a) Hours studied
b) Exam score
c) Student's age
d) Student's gender

**Answer:** b) Exam score

Let me know if you need any help with answers or explanations!

====================================================================================================

          +---------------------------+
          | Parallel<notes,quiz>Input |
          +---------------------------+
                ***             ***
              **                   **
            **                       **
+----------------+              +----------------+
| PromptTemplate |              | PromptTemplate |
+----------------+              +----------------+
          *                             *
          *                             *
          *                             *
  +------------+                  +----------+
  | ChatOpenAI |                  | ChatGroq |
  +------------+                  +----------+
          *                             *
          *                             *
          *                             *
+-----------------+            +-----------------+
| StrOutputParser |            | StrOutputParser |
+-----------------+            +-----------------+
                ***             ***
                   **         **
                     **     **
          +----------------------------+
          | Parallel<notes,quiz>Output |
          +----------------------------+
                         *
                         *
                         *
                +----------------+
                | PromptTemplate |
                +----------------+
                         *
                         *
                         *
                  +------------+
                  | ChatOpenAI |
                  +------------+
                         *
                         *
                         *
                +-----------------+
                | StrOutputParser |
                +-----------------+
                         *
                         *
                         *
            +-----------------------+
            | StrOutputParserOutput |
            +-----------------------+
"""