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
    

st.title("News Research Tool")

st.sidebar.title("News Articles URL")

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




if process_url_clicked :
    
    # load data from urls
    loader = UnstructuredURLLoader(urls=urls)

    # displayed on the screen when data is fetched.
    main_placefolder.write("Loading data from URLs...")

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

    # text splitting into chunks 
    splitted_text = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " "],
        chunk_size=500,  
    )

    # displayed on the screen when text is split into chunks.
    main_placefolder.write("Splitting text into chunks...")

    # creates a list called 'docs' which stores 3 further list and in which there are chunks in the form of string.
    # docs = [
    #     [  # Chunks from URL 1
    #         "First chunk of doc 1",
    #         "Second chunk of doc 1",
    #         "Third chunk of doc 1",
    #         ...
    #     ],
    #     [  # Chunks from URL 2
    #         "First chunk of doc 2",
    #         "Second chunk of doc 2",
    #         ...
    #     ],
    #     [  # Chunks from URL 3
    #         "First chunk of doc 3",
    #         "Second chunk of doc 3",
    #         ...
    #     ]
    # ]
    docs = []
    for i in range(len(data)) :
        docs.append(splitted_text.split_text(data[i].page_content))

    


    # converting this list of strings to list of documents because embeddings work on document format.
    document_format = []
    for i, doc in enumerate(docs):
        for chunk in doc :
            document_format.append(
                Document(
                    page_content=chunk,
                    metadata={"source": urls[0] if urls else f"doc_{i}"}  # use URL as source or a placeholder
                )
            )

    

   

    # creating faiss index. 
    dimension = len(embeddings.embed_query("hello world"))

    index = faiss.IndexFlatL2(dimension)

    # creating a store for the vector database
    vector_store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )

    uuids = []

    for _ in range(len(document_format)) :
        uuids.append(str(uuid4()))

    vector_store.add_documents(documents=document_format, ids=uuids)

    main_placefolder.write("Creating embeddings...")

    # saving the data locally into disc so that we can retrive it whenever we want. 
    vector_store.save_local("faiss_store")

query = main_placefolder.text_input("Enter your query")   



if query :
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

        

    
