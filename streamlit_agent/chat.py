from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAI
from langchain_openai import OpenAI
from langchain.chains import ConversationChain
from langchain_core.prompts import HumanMessagePromptTemplate,ChatPromptTemplate,MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage,SystemMessage


def chat(openai_key:str, chat_history:list, query:str):
    
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content="""You are a very intelligent digital AI system that understands humans properly. Your name is 42,
                you were named after the answer to the ultimate question in the hitch hikers guide to the galaxy.
                Answer the user queries accurately. use your knowledge base. Don't ever fail to provide a coding request assistance or 
                an assistance with writing a document like a resume or an official document because you were trained to know all of that.
                """
            ),  # The persistent system prompt
            MessagesPlaceholder(
                variable_name=chat_history
            ),  # Where the memory will be stored.
            HumanMessagePromptTemplate.from_template(
                "{human_input}"
            ),  # Where the human input will injected
        ]
        )

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    llm_chat = OpenAI(api_key=openai_key)

    llm_chain = LLMChain(
    llm=llm_chat,
    prompt=prompt,
    verbose=False,
    memory=memory,
)

    response = llm_chain.invoke(query)
    chat_history.append(response)
    return response['text'] 
