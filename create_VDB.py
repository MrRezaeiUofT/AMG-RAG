import os
import logging
from sentence_transformers import SentenceTransformer
from langchain.vectorstores import Chroma
import shutil  # For deleting directories

class SentenceTransformerEmbedding:
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name)

    def __call__(self, documents):
        # Make the class callable, so it works as an embedding function
        return self.embed_documents(documents)

    def embed_query(self, query):
        return self.model.encode(query, convert_to_tensor=True).tolist()

    def embed_documents(self, documents):
        return self.model.encode(documents, convert_to_tensor=True).tolist()


class MedicalQAChromaDB:
    def __init__(self, persist_directory="new_VDB/"):
        logging.basicConfig(filename='processing_log.log', level=logging.INFO)
        
        self.persist_directory = persist_directory
        # Initialize with a general-purpose model
        self.embedding_function = SentenceTransformerEmbedding('all-mpnet-base-v2')
        
        # Pass the embedding function to Chroma explicitly as a callable
        self.vectordb = Chroma(
            collection_name="medical_QA_books",
            persist_directory=persist_directory,
            embedding_function=self.embedding_function  # Pass the instance directly
        )
    

    def read_txt_files(self, directory):
        texts = {}
        for filename in os.listdir(directory):
            if filename.endswith(".txt"):
                with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
                    texts[filename] = file.read()
        return texts

    def split_text_into_chunks_with_overlap(self, text, chunk_size=512, overlap=100):
        words = text.split()
        chunks = []
        i = 0
        while i < len(words):
            chunk = ' '.join(words[i:i + chunk_size])
            chunks.append(chunk)
            i += chunk_size - overlap
        return chunks

    def store_embeddings_in_chroma(self, text_chunks, filename):
        self.vectordb.add_texts(
            texts=text_chunks,
            metadatas=[{"filename": filename, "chunk_id": i} for i in range(len(text_chunks))],
            ids=[f"{filename}_{i}" for i in range(len(text_chunks))]
        )
        print(f"Stored {len(text_chunks)} chunks for file: {filename}")

    def get_processed_files(self):
        if os.path.exists('processed_files.log'):
            with open('processed_files.log', 'r') as file:
                return set(file.read().splitlines())
        return set()

    def log_processed_file(self, filename):
        with open('processed_files.log', 'a') as file:
            file.write(f"{filename}\n")

    def ingest_files(self, directory, reset=False):
       
        texts = self.read_txt_files(directory)
        processed_files = self.get_processed_files()

        for filename, text in texts.items():
            if filename in processed_files:
                logging.info(f"Skipping already processed file: {filename}")
                continue

            logging.info(f"Processing file: {filename}")
            try:
                text_chunks = self.split_text_into_chunks_with_overlap(text)
                self.store_embeddings_in_chroma(text_chunks, filename)
                self.log_processed_file(filename)
                logging.info(f"Successfully processed file: {filename}")
            except Exception as e:
                logging.error(f"Error processing file {filename}: {e}")
                break

    def query_chroma(self, query_text, n_results=3):
        results = self.vectordb.similarity_search_with_score(query=query_text, k=n_results)
        return results

    def main(self, mode, directory=None, query_text=None, reset=False,n_results=5):
        if mode == "ingest" and directory:
            print(f"Ingesting files from directory: {directory}")
            self.ingest_files(directory, reset=reset)
            print("Ingestion complete.")
            result = ' '
        elif mode == "query" and query_text:
            # print(f"Querying vector DB with text: '{query_text}'")
            result = self.query_chroma(query_text,n_results=n_results)
            # for document in result:
            #     print(document)
        else:
            print("Invalid mode or missing arguments. Use 'ingest' with a directory or 'query' with a query text.")
            result=" "
        return result

# Example usage:
if __name__ == "__main__":
    db = MedicalQAChromaDB()

    # # Example of data ingestion with VDB reset
    db.main(mode="ingest", directory="dataset/data_clean/textbooks/en/", reset=True)

    # Example of querying
    aa=db.main(mode="query", query_text="symptoms of drug diabetes?",n_results=1)
    print(aa)