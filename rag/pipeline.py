from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from engine.sql_executor import analyze_sqlite_db
from utils import load_params


class Rag:
    def __init__(
        self,
        db_info: str,
        model_name: str = load_params("model_name"),
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        self.model_name = model_name
        self.embedding_model_name = embedding_model_name
        self.db_info = db_info

    def load_embedding_model(self) -> HuggingFaceEmbeddings:
        model_kwargs = {"device": "cpu"}
        encode_kwargs = {"normalize_embeddings": False}
        hf = HuggingFaceEmbeddings(
            model_name=self.embedding_model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
        )
        return hf

    def vectorize(self):
        hf = self.load_embedding_model()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
        )
        documents = text_splitter.create_documents([self.db_info])
        vector_db = FAISS.from_documents(documents, embedding=hf)
        return vector_db

    def llm_response(self, question, vector_store):
        llm = ChatOpenAI(model=self.model_name, base_url="https://api.groq.com/openai/v1")

        prompt = ChatPromptTemplate.from_template(
            """
            Based on the following database table information, determine the most relevant tables 
            for the user's question. Provide only the list of table names as your answer like this ['table1', 'table2', ...].
            <context>
            {context}
            </context>
            Question: {input}
            """
        )

        document_chain = create_stuff_documents_chain(llm, prompt)

        retriever = vector_store.as_retriever(search_kwargs={"k": 5})

        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        response = retrieval_chain.invoke({"input": question})
        return response


if __name__ == "__main__":
    pass
