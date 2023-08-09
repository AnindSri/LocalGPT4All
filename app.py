from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import GPT4All
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import streamlit as st
import os


load_dotenv()

model_path = os.environ.get('MODEL_PATH')
model_n_ctx = os.environ.get('MODEL_N_CTX')
persist_directory = os.environ.get('PERSIST_DIRECTORY')
embeddings_model_name = os.environ.get('EMBEDDINGS_MODEL_NAME')
model_n_batch = int(os.environ.get('MODEL_N_BATCH',8))
target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS',4))
chunk_size = int(os.environ.get('CHUNK_SIZE',500))
chunk_overlap = int(os.environ.get('CHUNK_OVERLAP',50))

st.header("Welcome")

@st.cache_data
def loading_splitting(new_doc):
    file_path = os.environ.get('FILE_PATH')

    with open(os.path.join("uploaded_docs", new_doc.name), "wb") as f:
        f.write(new_doc.getbuffer())

    file_path += '/' + str(new_doc.name)

    loader = PyPDFLoader(file_path=file_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    texts = text_splitter.split_documents(documents)

    return texts


def main():

    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    sb = st.sidebar
    page = sb.radio("", ["Personal AI bot", "Upload Pdf"])

    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})
    llm = GPT4All(model=model_path)        
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True, verbose=True)

    if page == "Personal AI bot":

        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])  

        if query := st.chat_input("Enter your question?"):
            st.chat_message("user").markdown(query)
            st.session_state.messages.append({"role": "user", "content": query})
            
            res = qa(query)
            answer = res['result']  
            with st.chat_message("assistant"):
                st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})

    else:
        st.subheader('Upload PDF')
        doc = st.file_uploader('Upload the document')

        if doc is not None:
            
            if os.path.isdir(persist_directory):
                print(f"Appending")
                db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
                collection = db.get()
                texts = loading_splitting(doc)       
                db.add_documents(texts)

                print(f"Append complete")
            
            else:
                print("Creating new")
                texts = loading_splitting(doc)
                db = Chroma.from_documents(texts, embeddings, persist_directory=persist_directory)
                print('New DB')
            db.persist()
            db = None


if __name__ == '__main__':
    main()