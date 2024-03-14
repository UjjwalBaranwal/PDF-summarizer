import streamlit as st
import pickle
from PyPDF2 import PdfReader
from dotenv import load_dotenv
# from pydantic import BaseModel, model_validator
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_openai import OpenAI
# from langchain.vectorstores import faiss
from langchain_community.vectorstores.faiss import FAISS
from langchain.chains.question_answering import load_qa_chain
import os
from langchain.callbacks import get_openai_callback
# from multiprocessing import Lock
# Sidebar contents
with st.sidebar:
    st.title('🤗💬 LLM Chat App')
    st.markdown('''
    ## About
    This app is an LLM-powered chatbot built using:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [OpenAI](https://platform.openai.com/docs/models) LLM model
 
    ''')
    add_vertical_space(5)
    st.write('Talk with your report')

load_dotenv()
# lock = Lock()
# st.write(os.getenv("OPENAI_API_KEY"))


def main():
    # st.write("hello this side ujjwal baranwal")
    st.header("Chat with PDF 💬")
    # upload a PDF file
    pdf = st.file_uploader("Upload your PDF", type='pdf')
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        st.write(pdf_reader)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        # st.write(text) ## this print all the text in the pdf
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        # convert the pdf text in different chunks
        chunks = text_splitter.split_text(text=text)
        # st.write(chunks)
        # embedding
        # storeName = pdf.name[:4]
        # # st.write(storeName)
        # if os.path.exists(f"{storeName}.pkl"):
        #     with open(f"{storeName}.pkl", "rb") as f:
        #         VectorStore = pickle.load(f)
        #     # st.write('Embeddings Loaded from the Disk')s
        # else:
        #     embeddings = OpenAIEmbeddings()
        #     VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
        #     # st.write(VectorStore[''])
        #     with open(f"{storeName}.pkl", "wb") as f:
        #         pickle.dump(VectorStore, f)
        #     st.write("created pickel file")
        embeddings = OpenAIEmbeddings()
        VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
        # Accept user questions/query
        query = st.text_input("Ask questions about your Report:")
        st.write(query)
        if query:
            docs = VectorStore.similarity_search(query=query, k=3)
            # st.write(docs)
            llm = OpenAI(temperature=0, tiktoken_model_name='gpt-3.5-turbo')
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            response = chain.run(input_documents=docs, question=query)
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=query)
                print(cb)
            st.write(response)


if __name__ == '__main__':
    main()
