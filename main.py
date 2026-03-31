import os 
import streamlit as st
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from uuid import uuid4
from langchain_classic.chains import RetrievalQAWithSourcesChain
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(model="llama-3.3-70b-versatile")
    

st.set_page_config(page_title="Research Tool", page_icon="📰")
st.title("Research Tool")

st.sidebar.title("Articles URL")

# 2. Use Session State for Persistence
if "processed" not in st.session_state:
    st.session_state.processed = False
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# st.empty() creates an “empty container” on the Streamlit page.
# We can use this container to display content later on.
main_placefolder = st.empty()


urls = []
for i in range(3) :
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)


# process_url_clicked holds boolean value :- 'TRUE' when button clicked; otherwise 'FALSE'
process_url_clicked = st.sidebar.button("Process URLs")


# uses model from huggingface for embedding
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")


# 4. Processing Logic

if process_url_clicked :
    
    if not any(urls):
        st.sidebar.error("Please enter at least one URL")
        st.stop()

    else :
        # displayed on the screen when data is fetched.
        main_placefolder.text("Data Loading... Started... ✅✅✅")

        # load data from urls

        loader = UnstructuredURLLoader(
            urls=urls, 
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}
            )
        data = loader.load()

        # data = [
        #     Document(
        #         page_content="This is the full content of page 1...",
        #         metadata={"source": "https://example.com/page1"}
        #     ),
        #     Document(
        #         page_content="This is the full content of page 2...",
        #         metadata={"source": "https://example.com/page2"}
        #     ),
        #     Document(
        #         page_content="This is the full content of page 3...",
        #         metadata={"source": "https://example.com/page3"}
        #     ),
        # ]
   

        main_placefolder.text("Splitting Data... Started... ✅✅✅")
    
        # text splitting into chunks 
        splitted_text = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", " "],
            chunk_size=1000, 
            chunk_overlap = 200 
        )   

        # displayed on the screen when text is split into chunks.
        main_placefolder.text("Splitting Data... Started... ✅✅✅")

        document_format = splitted_text.split_documents(data)  
    
        
        main_placefolder.text("Building Vector Store... Started... ✅✅✅")
        vector_store = FAISS.from_documents(document_format, embeddings)

        
        # save locally
        vector_store.save_local("faiss_store")

        # Store in session state
        st.session_state.vector_store = vector_store
        st.session_state.processed = True
        main_placefolder.text("Processing Complete! Ready for Q&A.")
    



query = st.text_input("Enter your query")   



if query :

    if st.session_state.processed and st.session_state.vector_store:
        

        if os.path.exists("faiss_store") :
            new_vector_store = FAISS.load_local(
            "faiss_store", embeddings, allow_dangerous_deserialization=True
        )

        chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=new_vector_store.as_retriever())
        result = chain({"question": query}, return_only_outputs=True)
    
        # result will be a dictionary 
        # {"answer" : "...", "sources" : "..."}

        st.header("Answer")
        st.subheader(result["answer"])

        

    
