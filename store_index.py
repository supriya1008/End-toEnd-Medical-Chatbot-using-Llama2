from src.helper import text_split, download_hugging_face_embeddings
from langchain.vectorstores import Pinecone
import pinecone
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from langchain.document_loaders import PyPDFLoader
import os

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_API_ENV = os.environ.get('PINECONE_ENV')

# print(PINECONE_API_KEY)
# print(PINECONE_API_ENV)


# Correct: Use PyPDFLoader for single PDF
loader = PyPDFLoader(r"C:\Users\supri\End-toEnd-Medical-Chatbot-using-Llama2\data\The-Gale-Encyclopedia-of-Medicine-3rd-Edition-staibabussalamsula.ac_.id_.pdf")
extracted_data = loader.load()
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()


import os
from pinecone import Pinecone, ServerlessSpec

pc = Pinecone(
      api_key=os.environ.get("PINECONE_API_KEY")
    )

index_name="medical-bot"


vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)