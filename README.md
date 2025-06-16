
# SQL_to_NLP

This repository contains the source code for the Kare Bot API, a chatbot designed to answer queries based on information retrieved from a SQL database using Retrieval-Augmented Generation (RAG).  The API is built using FastAPI and leverages Google's Gemini Pro model for natural language generation.

## Overview

The Kare Bot API provides a simple interface for interacting with a chatbot that can answer questions about a specific dataset stored in a SQL database.  It uses a RAG pipeline to retrieve relevant information from the database based on the user's query and then uses a large language model (LLM) to generate a natural language response.  Few-shot learning is employed via a Chroma vector database to improve the quality of the SQL query generation.

## Features

*   **RAG Pipeline:** Implements a Retrieval-Augmented Generation pipeline for answering user queries.
*   **SQL Database Integration:** Connects to a SQL database to retrieve relevant information.
*   **Google Gemini Pro:** Utilizes Google's Gemini Pro model for natural language generation.
*   **FastAPI:** Built using the FastAPI framework for high performance and ease of use.
*   **Few-Shot Learning:** Uses a Chroma vector database to store and retrieve examples for few-shot learning, enhancing SQL query generation accuracy.
*   **Semantic Similarity Example Selection:** Employs semantic similarity to select the most relevant examples from the Chroma vector database.
*   **Environment Variable Configuration:** Relies on environment variables for sensitive information like API keys and database URLs.
*   **Logging:** Implements detailed logging for debugging and monitoring.
*   **Date/Time Handling:** Includes sophisticated date/time handling to allow users to ask questions relative to today.
*   **Clear Error Handling:**  Provides user-friendly error messages when data is not available or issues occur.