from langchain import LLMChain, FewShotPromptTemplate, PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.prompts.example_selector import LengthBasedExampleSelector

examples = [
    {
        "query": "How do I become a better programmer?",
        "answer": "Try talking to a rubber duck; it works wonders."
    }, {
        "query": "Why is the sky blue?",
        "answer": "It's nature's way of preventing eye strain."
    }
]

example_template = """
User: {query}
AI: {answer}
"""

prefix = """The following are excerpts from conversations with an AI
assistant. The assistant is typically sarcastic and witty, producing
creative and funny responses to users' questions. Here are some
examples: 
"""

suffix = """
User: {query}
AI: """

example_prompt = PromptTemplate(
    input_variables=["query", "answer"],
    template=example_template
)

example_selector = LengthBasedExampleSelector(
    examples=examples,
    example_prompt=example_prompt,
    max_length=100  
)

dynamic_prompt_template = FewShotPromptTemplate(
    example_selector=example_selector, 
    example_prompt=example_prompt,
    prefix=prefix,
    suffix=suffix,
    input_variables=["query"],
    example_separator="\n"
)

llm = ChatOpenAI(model_name="gpt-3.5-turbo")

chain = LLMChain(llm=llm, prompt=dynamic_prompt_template)

# Run the LLMChain with input_data
input_data = {"query": "Who invented the telephone?"}
response = chain.run(input_data)

print(response)