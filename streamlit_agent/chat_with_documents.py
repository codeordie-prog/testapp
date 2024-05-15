import os
import tempfile
import chat
import streamlit as st
from io import BytesIO
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader,TextLoader,CSVLoader
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import (
    StreamlitChatMessageHistory,
)
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import LLMChain
from langchain_core.prompts import HumanMessagePromptTemplate,ChatPromptTemplate,MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage,SystemMessage
from langchain_core.runnables.history import RunnableWithMessageHistory #for chain with history

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

st.set_page_config(page_title="Ask Fortytwo", page_icon="👽", layout="centered")
st.title("👽 Ask Fortytwo 🛸🌌")
st.markdown("*Unlocking the mysteries of the universe, one question at a time*")

# Explanation of the App
st.header('About the App')
st.write("""
42, named after the answer to the ultimate question of life in the HitchHiker's Guide to the Galaxy, is an advanced question-answering platform that allows users to upload documents in the formats (pdf,txt and csv) and receive answers to
their queries based on the content of these documents. 
Utilizing RAG approach powered by OpenAI's GPT models, the app provides insightful and contextually relevant answers.

### How It Works
- add your secret openAI API key on the top left slider.
- for a basic chat without document query, use the chat_with_42 query entry
         
- for document query, upload a Document: You can upload any document in `.pdf`,`.txt`, or `.csv` format.
- you can also upload multiple documents and query them all together.
- Ask a Question: After uploading the document, type in your question related to the document's content.
- Get Answers: AI analyzes the document and provides answers based on the information contained in it.
         
- Note : clear message history button resets the context of the conversation else, 42 might answer 'I don't know".

""")

#instructions
# Instructions for getting an OpenAI API key
st.subheader("Get an OpenAI API key")
st.write("You can get your own OpenAI API key by following the instructions:")
st.write("""
1. Create an openAI account
2. Go to [OpenAI API Keys](https://platform.openai.com/account/api-keys).
3. Click on the `+ Create new secret key` button.
4. Next, enter an identifier name (optional) and click on the `Create secret key` button.
""")

st.write("""
### GET STARTED

simply upload your documents and query 42, or just engage in a chat below:
""")
try:
    
    class StreamHandler(BaseCallbackHandler):
        def __init__(self, container: st.delta_generator.DeltaGenerator, initial_text: str = ""):
            self.container = container
            self.text = initial_text
            self.run_id_ignore_token = None

        def on_llm_start(self, serialized: dict, prompts: list, **kwargs):
            # Workaround to prevent showing the rephrased question as output
            if prompts[0].startswith("Human"):
                self.run_id_ignore_token = kwargs.get("run_id")

        def on_llm_new_token(self, token: str, **kwargs) -> None:
            if self.run_id_ignore_token == kwargs.get("run_id", False):
                return
            self.text += token
            self.container.markdown(self.text)


    class PrintRetrievalHandler(BaseCallbackHandler):
        def __init__(self, container):
            self.status = container.status("**Context Retrieval**")

        def on_retriever_start(self, serialized: dict, query: str, **kwargs):
            self.status.write(f"**Question:** {query}")
            self.status.update(label=f"**Context Retrieval:** {query}")
        def on_retriever_end(self, documents, **kwargs):
            for idx, doc in enumerate(documents):
                source = os.path.basename(doc.metadata["source"])
                self.status.write(f"**Document {idx} from {source}**")
                self.status.markdown(doc.page_content)
            self.status.update(state="complete")


    #the interaction setup part
    openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()

    uploaded_files = st.sidebar.file_uploader(
            label="Upload files", type=["pdf","txt","csv"], accept_multiple_files=True
        )

    #chat setup
    # Setup LLM and QA chain - msg variable for chat history from streamlitchatmessagehistory
    #set up the memory with chat_memory as the msg variable -use conversational buffer memory
    #set up the prompt
    #initialize the llm with streaming true
    #initialize the chain with all the set up fields i.e promp,memory,verbose false and llm
    #use the chain to invoke chat query

    #function-1   
    @st.cache_resource(ttl="2h")
    def configure_retriever(uploaded_files):
        # Read documents
        docs = []
        temp_dir = tempfile.TemporaryDirectory()
        for file in uploaded_files:
            temp_filepath = os.path.join(temp_dir.name, file.name)
            st.write(temp_filepath)
            with open(temp_filepath, "wb") as f:
                f.write(file.getvalue())

            #load pdf,txt and csv
            if temp_filepath.endswith(".pdf"):
                loader = PyPDFLoader(temp_filepath)
                docs.extend(loader.load())
                
            elif temp_filepath.endswith(".txt"):
                loader = TextLoader(temp_filepath)
                docs.extend(loader.load())
            
            elif temp_filepath.endswith(".csv"):
                loader = CSVLoader(temp_filepath)
                docs.extend(loader.load())   

        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)

        # Create embeddings and store in vectordb
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectordb = DocArrayInMemorySearch.from_documents(splits, embeddings)

        # Define retriever
        retriever = vectordb.as_retriever() #retrieve default

        return retriever

    #function-2
    #define download txt
    def create_and_download(text_content):
        """Generates a text file in memory and offers a download button."""
        # Create a BytesIO object to store the content
        buffer = BytesIO()

        # Write the content to the buffer
        if isinstance(text_content, bytes):
            buffer.write(text_content)
        else:
            buffer.write(text_content.encode('utf-8'))

        buffer.seek(0)

        # Provide download button
        st.download_button(
            label="Download Text File",
            data=buffer,
            file_name="my_text_file.txt",
            mime="text/plain"
        )
    
    #function-3 chat session

    st.cache_resource(ttl="2hr")
    def chat_with_42():
        msgs2 = StreamlitChatMessageHistory(key="chat_history")
        memory2 = ConversationBufferMemory(memory_key="chat_history", chat_memory=msgs2, return_messages=True)
        llm2 = ChatOpenAI(
            model_name="gpt-3.5-turbo", openai_api_key=openai_api_key, temperature=0, streaming=True
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(
                    content="""You are a very intelligent digital AI system that understands humans properly. Your name is 42,
                    you were named after the answer to the ultimate question in the hitch hikers guide to the galaxy. Your were created by Kelvin Ndeti,
                    in association with Dr. Whbet Paulos, inspired by the need to utilize Retrieval Augmented Generation in data quering.
                    Answer the user queries accurately. use your knowledge base. Don't ever fail to provide a coding request assistance or 
                    an assistance with writing a document like a resume or an official document because you were trained to know all of that.
                    """
                ),  # The persistent system prompt
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{question}"),  # Where the memory will be stored.
            
            ]
            )
        
        llm_chain = LLMChain(
        llm=llm2,
        verbose=False,
        memory=memory2,
        prompt=prompt
        )

        #chain with history
        chain_with_history = RunnableWithMessageHistory(
        llm_chain,
        lambda session_id: msgs2,  # Always return the instance created earlier
        input_messages_key="question",
        history_messages_key="chat_history"
        )

        if len(msgs2.messages) == 0 or st.sidebar.button("Clear chat_with_42 message history"):
            msgs2.clear()
            msgs2.add_ai_message("Hey carbon entity, lets talk!")

        if chat_query := st.text_input("Chat with 42, let's chat. enter query : "):

            for msg in msgs2.messages:
                st.chat_message(msg.type).write(msg.content)

            if prompt := st.chat_input():
                st.chat_message("human").write(prompt)

            #configure session id
            config = {"configurable": {"session_id": "any"},}
            response = chain_with_history.invoke({"question" : chat_query},config=config)
            # response = llm_chain.invoke(chat_query)
            st.write("response: ",response["text"])

            #download button
            if st.button("Create and download txt"):
                create_and_download(text_content=response['text'])

     #function-4 query documents           
    def query_documents():
       
        if not uploaded_files:
            st.info("Please upload documents to continue.")
            st.stop()

        system_prompt = (
            "Use the given context to answer the question. "
            "If you don't know the answer, say you don't know. "
            "Use three sentence maximum and keep the answer concise. "
            "Context: {context}"
        )
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{input}"),
            ]
        )
        
        retriever = configure_retriever(uploaded_files)

        # Setup memory for contextual conversation for the documents part
        msgs = StreamlitChatMessageHistory()
        memory = ConversationBufferMemory(memory_key="chat_history", chat_memory=msgs, return_messages=True)

        # Setup LLM and QA chain for the documents part
        llm = ChatOpenAI(
            model_name="gpt-4", openai_api_key=openai_api_key, temperature=0, streaming=True
        )

        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        chain = create_retrieval_chain(retriever, question_answer_chain)

        qa_chain = ConversationalRetrievalChain.from_llm(
            llm, retriever=retriever, memory=memory, verbose=True
        )

        if len(msgs.messages) == 0 or st.sidebar.button("Clear message history"):
            msgs.clear()
            msgs.add_ai_message("Hey carbon entity, Want to query your documents? ask me!")

        avatars = {"human": "user", "ai": "assistant"}
        for msg in msgs.messages:
            st.chat_message(avatars[msg.type]).write(msg.content)
        
        st.markdown("Document query section. Utilize RAG you curious being.")
        if user_query := st.chat_input(placeholder="Ask me about  your documents!"):
            st.chat_message("user").write(user_query)

            with st.chat_message("ai"):
                retrieval_handler = PrintRetrievalHandler(st.container())
                stream_handler = StreamHandler(st.empty())
                #qa_chain.run(user_query, callbacks=[retrieval_handler, stream_handler])
                chain.invoke({"input":user_query})
               
    #main function
    def main():

        if uploaded_files:
        
            query_documents()
        else:
            chat_with_42()

    
    #call main
    main()


except Exception as e:
    st.write("an error occured check the key",e)
