# 🤖 Kare Bot — Natural Language Faculty Search

> An AI-powered system that transforms university timetable PDFs into a searchable knowledge base, allowing students to query faculty availability using natural language.

## 🚀 The Problem

Faculty schedules were distributed across multiple timetable PDFs.

Finding answers to questions such as:

* "When is Dr. Kumar free today?"
* "Where can I meet my DBMS faculty tomorrow?"
* "Who is teaching in Lab 3 right now?"

required manually searching through numerous timetable documents.

## 💡 The Solution

Kare Bot automatically converts timetable PDFs into a structured relational database and enables natural language search over that data.

```text
Timetable PDFs
       ↓
PDF Parsing & Information Extraction
       ↓
Relational Database Generation
       ↓
Natural Language Query
       ↓
LLM → SQL Translation
       ↓
Database Retrieval
       ↓
Human-Friendly Response
```

## ✨ Key Features

* 📄 Automated timetable PDF parsing
* 🏗️ Dynamic SQL schema generation
* 🤖 Natural Language → SQL conversion
* 🧠 Semantic few-shot example retrieval using ChromaDB
* 🔍 Retrieval-Augmented Query Generation
* 📅 Relative date understanding (Today, Tomorrow, Day After Tomorrow)
* ⚡ Faculty availability and timetable search
* 🔐 User authentication and access control

## 🛠️ Tech Stack

**Python • LangChain • Gemini • ChromaDB • MySQL • SQLAlchemy • Streamlit**

## 🎯 Impact

Instead of manually searching through dozens of timetable PDFs, students can simply ask:

> "When is my faculty available?"

and receive an instant answer generated from structured timetable data.

This project was built to learn LLMs, LangChain, RAG pipelines, vector databases, and Text-to-SQL systems while solving a real problem faced by students on campus.
