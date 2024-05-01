from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAI
from langchain_openai import OpenAI
from langchain.chains import ConversationChain


def chat(openai_key:str):
    template = """You are a funny chatbot called 42. 
    You are named after the Hitch Hikers Guide to the Galaxy famous 42. having a conversation with a human.

    {chat_history}
    Human: {human_input}
    Chatbot:"""

    prompt = PromptTemplate(
        input_variables=["chat_history", "human_input"], template=template
    )
    memory = ConversationBufferMemory(memory_key="chat_history",return_messages=True)

    llm_chat = OpenAI(api_key=openai_key)


    llm = OpenAI(temperature=0,api_key=openai_key)

    conversation = ConversationChain(
    llm=llm,
    prompt = prompt,
    verbose=True,
    memory=ConversationBufferMemory()
)

    return conversation
