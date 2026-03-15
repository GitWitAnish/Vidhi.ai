<p align="center">
  <img src="assets/images/logo.jpeg" alt="VIDHI>AI" width="150" height="150" style="border-radius: 20px;">
</p>

<h1 align="center">VIDHI.AI</h1>

<p align="center">
  <strong>AI-Powered Legal Assistant for Nepal</strong><br>
  <em>Making legal knowledge and government navigation accessible to every Nepali citizen.</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/LLaMA_3.1-8B-purple?logo=meta" alt="LLaMA">
  <img src="https://img.shields.io/badge/FAISS-Vector_DB-green" alt="FAISS">
  <img src="https://img.shields.io/badge/React-18-61DAFB?logo=react" alt="React">
  <img src="https://img.shields.io/badge/FastAPI-0.100+-009688?logo=fastapi" alt="FastAPI">
 
</p>

---

##  Problem Statement

 Laws are easily accessible **but not easily understandable.**

---

##  Our Solution

**VIDHI.AI** is a conversational AI that lets anyone ask legal questions in plain language and get accurate, cited answers from **35+ Nepali laws and acts** and also custom navigation for Pokhara valley.

## How It Works

<p align="center">
  <img src="assets/images/workflow.png" alt="How it works" width="800">
</p>

---

## Key Features

| Feature                     | Description                                       |
| --------------------------- | ------------------------------------------------- |
| 1. **Natural Language Q&A** | Ask in English or Nepali, get clear answers       |
| 2. **35+ Legal Documents**  | Constitution, Civil Code, Criminal Code, and more |
| 3. **Source Citations**     | Every answer backed by actual legal text          |
| 4. **Procedure Navigation** | Step-by-step guidance for government services     |
| 5. **Voice Support**        | Speak your question, hear the answer              |
| 6. **Fast Retrieval**       | FAISS-powered vector search in milliseconds       |

---

## Tech Stack

| Layer          | Technology                           |
| -------------- | ------------------------------------ |
| **LLM**        | Meta LLaMA 3.1 8B (via HuggingFace)  |
| **Embeddings** | SentenceTransformers (MiniLM-L6)     |
| **Vector DB**  | FAISS                                |
| **Backend**    | FastAPI                              |
| **Frontend**   | React 18 + Vite / Streamlit          |
| **Voice**      | Edge TTS + Google Speech Recognition |

---

---

## Architecture

```
nyaya-exe/
├── streamlit/          #   Alternative UI
├── frontend/           #   React frontend
├── backend/            #   FastAPI server
├── scripts/            #   Core RAG logic
│   ├── embedding.py    #   Text embeddings
│   ├── vector.py       #   FAISS operations
│   ├── llm_wrapper.py  #   LLaMA integration
│   └── voice.py        #   TTS/STT
├── database/           #   Vector index
└── dataset/            #   Legal documents
```

## Future Roadmap

- Full Nepali language support
- Analytics dashboard
- User authentication
- Mobile app (React Native)
- Integration with Nepal Law Commission
- Real-time document updates
- Lawyer consultation matching
- Document generation (legal templates)
- Multi-jurisdictional support

---

## Team

<table>
  <tr>
    <td align="center">
      <strong>Milan Bastola</strong><br>
      <em>Full Stack Developer</em><br>
      <a href="https://github.com/Milan342">GitHub</a>
    </td>
    <td align="center">
      <strong>Prasanna Pahari</strong><br>
      <em>ML/AI Engineer</em><br>
      <a href="https://github.com/hopelessxD">GitHub</a>
    </td>
    <td align="center">
      <strong>Anish Karki</strong><br>
      <em>ML/AI Engineer</em><br>
      <a href="https://github.com/GitWitAnish">GitHub</a>
    </td>
  </tr>
</table>

---

## Acknowledgments

- **Nepal Law Commission** - For making legal documents publicly available
- **Meta AI** - For the open-source LLaMA 3.1 model
- **HuggingFace** - For model hosting and inference API
- **FAISS Team** - For the efficient vector similarity search library
- **Sentence Transformers** - For pre-trained embedding models
- **Streamlit** - For the amazing data app framework
- **Open Source Community** - For all the libraries that made this possible

### Resources & References

- [Nepal Law Commission](https://lawcommission.gov.np/)
- [LLaMA Documentation](https://llama.meta.com/)
- [FAISS Documentation](https://faiss.ai/)

---


<p align="center">
  <strong> Justice is everyone's right. (न्याय सबैको हक हो)</strong>
</p>

<p align="center">
  Made with ❤️ and Passion
</p>
