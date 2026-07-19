import chromadb
from chromadb import Collection, QueryResult
from itertools import islice


class ChromaDBClient:
    def __init__(self, path="./chroma_storage"):
        self.client = chromadb.PersistentClient(path=path)


    def create_collection(self, collection_name: str, 
                        description: str,
                        embedding_function = None,
                        ) -> Collection:
        return self.client.create_collection(
            name=collection_name, 
            embedding_function=embedding_function,
            metadata={
                "description": description
            })


    def batched(self, iterable, batch_size):
        for i in range(0, len(iterable), batch_size):
            yield iterable[i:i + batch_size]


    def add_documents(self, collection_name, ids, embeddings, metadata=None):
        collection = self.client.get_collection(collection_name)

        max_batch_size = 5000  # меньше лимита 5461

        if metadata is None:
            for ids_batch, emb_batch in zip(
                self.batched(ids, max_batch_size),
                self.batched(embeddings, max_batch_size)
            ):
                collection.add(
                    ids=ids_batch,
                    embeddings=emb_batch,
                )
        else:
            for ids_batch, emb_batch, meta_batch in zip(
                self.batched(ids, max_batch_size),
                self.batched(embeddings, max_batch_size),
                self.batched(metadata, max_batch_size)
            ):
                collection.add(
                    ids=ids_batch,
                    embeddings=emb_batch,
                    metadatas=meta_batch,
                )
                
    def get_closest_docs(self, collection_name, query_embeddings, n_closest=10) -> QueryResult:
        return self.client.get_collection(collection_name).query(
                query_embeddings=query_embeddings,
                n_results=n_closest,
                include=["embeddings", "metadatas", "distances"]
        )


    def delete_collection(self, collection_name):
        self.client.delete_collection(collection_name)