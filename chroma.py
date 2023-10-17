from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.vectorstores import Chroma
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv, find_dotenv
import os
import openai
import tiktoken

_=load_dotenv(find_dotenv())
openai.api_key = os.environ["OPENAI_API_KEY"]

directory = 'bankpdf'

def load_docs(directory):
  loader = DirectoryLoader(directory)
  documents = loader.load()
  return documents

documents = load_docs(directory)

print("Documents Loaded !\n")

def count_token(text):
    encoding = tiktoken.get_encoding("cl100k_base")
    #encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    num_token = len(encoding.encode(text))
    return num_token

def split_docs(documents,chunk_size=1280,chunk_overlap=256):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap,separators = "\n",length_function = count_token)
    docs = text_splitter.split_documents(documents)
    return docs

docs = split_docs(documents)

print("Chunking Done !\n")

print("Number of Chunks: ",len(docs), "\n")

embeddings = OpenAIEmbeddings()

#Creating Vector Store with Chroma DB

db3 = Chroma.from_documents(docs, embeddings)

print("Vector store created !\n")

persist_directory = "chroma_db"

vectordb = Chroma.from_documents(
    documents=docs, embedding=embeddings, persist_directory=persist_directory)

vectordb.persist()

model_name = "gpt-3.5-turbo"

llm = ChatOpenAI(openai_api_key=openai.api_key, model_name=model_name)

memory = ConversationBufferMemory(

memory_key='chat_history', return_messages=True)

# load from disk

db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

retriever = db.as_retriever()

print("Loaded in retriever !\n")

#

qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

answer = qa.run("What is the annual fee for the SBI Card ?")

print(answer)

