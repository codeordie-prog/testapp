from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAI
from langchain_openai import OpenAI
from langchain.chains import ConversationChain
from langchain_core.prompts import HumanMessagePromptTemplate,ChatPromptTemplate
from langchain_core.messages import AIMessage, HumanMessage,SystemMessage


def chat(openai_key:str):
    template = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content=(
                "You are a smart assistant called 42. answer all queries accurately "
                
            )
        ),
        HumanMessagePromptTemplate.from_template("{text}"),
    ]
)


    prompt = PromptTemplate(
       template=template
    )
    memory = ConversationBufferMemory(memory_key="chat_history")

    llm_chat = OpenAI(api_key=openai_key)

    llm_chain = LLMChain(
    llm=llm_chat,
    prompt=prompt,
    verbose=False,
    memory=memory,
)

    return llm_chain
