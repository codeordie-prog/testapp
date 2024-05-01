import os
import tempfile
import chat
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader,TextLoader,CSVLoader
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.text_splitter import RecursiveCharacterTextSplitter

st.set_page_config(page_title="Ask Fortytwo", page_icon="👽", layout="centered")
st.title("👽 Ask Fortytwo")
st.markdown("*Unlocking the mysteries of the universe, one question at a time*")

# Explanation of the App
st.header('About the App')
st.write("""
42, named after the answer to the ultimate question of life, is an advanced question-answering platform that allows users to upload documents in the formats (pdf,txt and csv) and receive answers to
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


### Get Started
Simply upload your document and start asking questions!
""")


try:
        
    @st.cache_resource(ttl="1h")
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



    openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()


    def chat_session(interaction_count : int, chat_history = []):

        if chat_query := st.text_input("Chat with 42, enter query : "):
            interaction_count += 1
            if interaction_count > 20:
                st.write("Maximum of 20 interactions reached. Please reset the conversation.")
                if st.button("Reset"):
                    chat_session(interaction_count=0,chat_history=[])
                else:
                    st.stop()

            llm_ch = chat.chat(openai_key=openai_api_key,chat_history=chat_history)
            response = llm_ch.invoke(chat_query)
            chat_history.append(response)
            st.write("response: ",response['text'])
            chat_session(interaction_count)
    
    chat_session(20)


    uploaded_files = st.sidebar.file_uploader(
        label="Upload files", type=["pdf","txt","csv"], accept_multiple_files=True
    )
    if not uploaded_files:
        st.info("Please upload documents to continue.")
        st.stop()


    retriever = configure_retriever(uploaded_files)

    # Setup memory for contextual conversation
    msgs = StreamlitChatMessageHistory()
    memory = ConversationBufferMemory(memory_key="chat_history", chat_memory=msgs, return_messages=True)

    # Setup LLM and QA chain
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo", openai_api_key=openai_api_key, temperature=0, streaming=True
    )
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm, retriever=retriever, memory=memory, verbose=True
    )

    if len(msgs.messages) == 0 or st.sidebar.button("Clear message history"):
        msgs.clear()
        msgs.add_ai_message("Hey carbon entity, Want to query your documents? ask me!")

    avatars = {"human": "user", "ai": "assistant"}
    for msg in msgs.messages:
        st.chat_message(avatars[msg.type]).write(msg.content)

    if user_query := st.chat_input(placeholder="Ask me about  your documents!"):
        st.chat_message("user").write(user_query)

        with st.chat_message("assistant"):
            retrieval_handler = PrintRetrievalHandler(st.container())
            stream_handler = StreamHandler(st.empty())
            response = qa_chain.run(user_query, callbacks=[retrieval_handler, stream_handler])


except Exception as e:
    st.write("an error occured check the key")
