import hashlib
import json
import os
from functools import lru_cache

from langchain.docstore.document import Document
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS

from utils import load_params


class Rag:
    def __init__(
        self,
        db_info_path: str,
        model_name: str = load_params("model_name"),
        embedding_model_name: str = "BAAI/bge-small-en-v1.5",
    ):
        self.model_name: str = model_name
        self.embedding_model_name: str = embedding_model_name
        self.embedding_model: FastEmbedEmbeddings = self.load_embedding_model()
        self.db_info_path: str = db_info_path
        self.docs: list = self.load_db_info()
        print("cache status: ", self.load_db_info.cache_info())

    @lru_cache(maxsize=1)
    def load_db_info(self) -> list:
        with open(self.db_info_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        docs = []
        for table_name, table_info in data.items():
            doc_content = (
                f"Table: {table_name}\n"
                f"Description: {table_info.get('description', '')}\n"
                f"Sample Questions: {table_info.get('sample_questions', [])}\n"
                f"Columns:\n{json.dumps(table_info.get('columns', {}), indent=4)}"
            )
            docs.append(
                Document(page_content=doc_content, metadata={"table": table_name})
            )
        return docs

    def load_embedding_model(self) -> FastEmbedEmbeddings:
        embeddings = FastEmbedEmbeddings(
            model_name=self.embedding_model_name, cache_dir="rag/embeddings"
        )
        return embeddings

    def compute_db_hash(self) -> str:
        with open(self.db_info_path, "rb") as f:
            data = f.read()
        return hashlib.md5(data).hexdigest()

    def vectorize(self):
        save_path = "vector_store"
        db_hash = self.compute_db_hash()
        hash_file = os.path.join(save_path, "hash.txt")

        if os.path.exists(save_path) and os.path.exists(hash_file):
            with open(hash_file, "r") as f:
                stored_hash = f.read().strip()
            if stored_hash == db_hash:
                print("Existing vector store is up-to-date. Skipping vectorization.")
                return

        vector_store = FAISS.from_documents(self.docs, self.embedding_model)
        os.makedirs(save_path, exist_ok=True)
        vector_store.save_local(save_path)
        with open(hash_file, "w") as f:
            f.write(db_hash)
        print(f"Vector store saved at: {save_path}")

    def vector_search(self, question, vector_store, top_k=3):
        results = vector_store.similarity_search_with_score(question, k=top_k)
        return results


if __name__ == "__main__":
    db_info_path = "database_info.json"

    rag = Rag(db_info_path=db_info_path)

    rag.vectorize()

    vector_store = FAISS.load_local(
        "vector_store", rag.embedding_model, allow_dangerous_deserialization=True
    )

    sample_question = "find name of people who is actor and director alsolimit by 5"

    search_results = rag.vector_search(sample_question, vector_store, top_k=3)

    for doc, score in search_results:
        print(f"Score: {score:.4f}")
        print(doc.page_content)
        print("-" * 80)
