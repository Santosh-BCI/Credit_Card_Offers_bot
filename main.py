import openai
from langchain.agents.agent_types import AgentType
from langchain.sql_database import SQLDatabase
from langchain.llms.openai import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv, find_dotenv
import os
import openai
import pandas as pd
import sqlite3
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA



import numpy as np 
from pandasai import PandasAI 
from pandasai import SmartDataframe
from pandasai.llm.openai import OpenAI

_=load_dotenv(find_dotenv())
openai.api_key = os.environ["OPENAI_API_KEY"]

def sql_function(user_input):

    # Connect to an existing database file
    conn = sqlite3.connect('rr.db')

    # Create a cursor object to execute SQL queries
    cursor = conn.cursor()

    # Execute a SELECT query
    cursor.execute('SELECT * FROM offers')

    # Fetch the query results
    rows = cursor.fetchall()

    # print("Table Data: ",rows)

    # define the SQL query
    query = "SELECT * FROM offers"

    # execute the query and return the results as a Pandas dataframe
    df = pd.read_sql(query, conn)

    # print("The Dataframe contains: \n", df)

    #initiating Pandas AI
    llm = OpenAI(api_token=openai.api_key) 
    pandas_ai = PandasAI(llm, conversational=False)

    sdf = SmartDataframe(df, config={"llm": llm, "verbose": False})

    return sdf.chat(user_input)

def chroma_function(user_input):
    # Set up OpenAI embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=openai.api_key)

    # Load your persisted vector store
    persist_directory = "chroma_db"
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

    # Set up retriever
    retriever = db.as_retriever()

    # Set up your model and memory
    model_name = "gpt-3.5-turbo-16k"
    llm = ChatOpenAI(openai_api_key=openai.api_key, model_name=model_name)
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

    # Set up QA model
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", memory = memory, retriever=retriever)

    answer = qa.run(user_input)

    return answer

def bot_for_interaction(conversation):
    response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo-16k",
    messages=conversation
    )

    bot_response = response["choices"][0]["message"]["content"]
    return bot_response.strip()

while True:
    user_input = input("User: ")

    # matching_docs = db.similarity_search(user_input, k=1)

    if user_input.lower() in ["exit", "quit", "bye"]:
        print("Have a good day!")
        break

    # qa = ConversationalRetrievalChain.from_llm(llm=llm, memory=memory,chain_type="stuff", retriever=db.as_retriever())
    credit_card_offers = chroma_function(user_input) #from Chroma DB

    flight_offers = sql_function(user_input) #from SQL DB

    print("\n")

    print("flight_offers: ", flight_offers)

    bot_instructions = f"""Your task is to answer user questions in the following manner:

    Rule - 1: If you find user query is related to flight booking then answer from ```{flight_offers}``` only. For this case follow the following instructions:
    ```
    - Reply in points, like discount percent and maximum allowed discount in one line, coupon code in another line, Expiry date in another line etc.
    - Do not reply in tabular form. 
    - If you get multiple offers in ```{flight_offers}```, matching the user input, then reply user with all the offers, don't skip any offer.

    ```

    Rule - 2: For all the other cases, answer user question from ```{credit_card_offers}``` only.

    """

    print("Credit_Card_Offers: ", credit_card_offers)

    conversation = [
        {"role": "system", "content": bot_instructions}
    ]

    bot_response = bot_for_interaction(conversation + [{"role": "user", "content": user_input}])

    print("Bot: ", bot_response)
    print("\n")
