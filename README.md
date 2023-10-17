# Credit_Card_Offers_bot
This Python script demonstrates the creation of an AI-powered conversational bot that can provide information and answers to user queries from multiple sources. The bot is designed to respond to user input related to flight offers on credit card and credit card related information. It uses OpenAI's GPT-3.5 Turbo model for generating responses and interacts with various data sources, such as SQL databases and persisted vector stores.

### Prerequisites

Before you can run this code, you need to make sure you have the following requirements installed:
Python 3.x
OpenAI Python Library
Pandas
SQLite3
Langchain and its dependencies
An OpenAI API key, which you can obtain by signing up on the OpenAI platform

### Note
### Before executing main.py, first run chroma.py and sql.py

The bot will continuously prompt you for user input. Enter your query or question.

The bot will generate responses based on the type of query. If the query is related to flight offers on credit card, it will retrieve information from an SQL database. If the query is related to credit card information, it will use a persisted vector store (Chroma) to provide answers.

The bot will follow the provided rules for formatting and responding to user queries.

To exit the bot, you can enter "exit," "quit," or "bye" when prompted for user input.
