from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAI
from langchain_openai import OpenAI
from langchain.chains import ConversationChain


def chat(openai_key:str):
    template = """You are a helpful smart assistant called 42. 
    You are named after the Hitch Hikers Guide to the Galaxy famous 42. answer all queries accurately and provide detailed explanation if needed.

    {chat_history}
    Human: {human_input}
    Assistant:"""

    prompt = PromptTemplate(
        input_variables=["chat_history", "human_input"], template=template
    )
    memory = ConversationBufferMemory(memory_key="chat_history",return_messages=False)

    llm_chat = OpenAI(api_key=openai_key)

    llm_chain = LLMChain(
    llm=llm_chat,
    prompt=prompt,
    verbose=False,
    memory=memory,
)

    return llm_chain
