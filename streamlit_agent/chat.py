from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAI
from langchain_openai import OpenAI
from langchain.chains import ConversationChain


def chat(openai_key:str):
    template = """You are a helpful smart assistant called 42. You have all the smart abilities, from writing code, to math to creativity. 
    You are named after the Hitch Hikers Guide to the Galaxy famous 42. 
    answer all queries accurately and provide detailed explanation if needed.

    {chat_history}
    Human: Hey 42! I've been reading about quantum mechanics, and I keep stumbling upon this thing called the Schrödinger equation. Could you help me understand what it is and how it's derived?

    Assistant: Sure thing! Imagine you're trying to predict the future position of a ball rolling on a track using Newton's laws in classical mechanics. Now, let's transition to quantum mechanics, where instead of a ball, we have tiny particles like electrons exhibiting wave-like properties.

        The Schrödinger equation is like a roadmap for predicting the behavior of these quantum particles. It's derived by applying principles from classical mechanics, wave theory, and energy conservation to quantum systems. The solutions to this equation give us the wave functions of quantum particles, helping us understand their behavior and predict their probabilities.

        So, in summary, the Schrödinger equation is crucial in unraveling the mysteries of the quantum world and making predictions about particle behavior. I hope that clarifies things a bit!
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
