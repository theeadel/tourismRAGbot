# tourismRAGbot
touristRAGbot is a smart, context-aware chatbot designed to recommend adventure and travel activities across Lebanon. It leverages a combination of LLMs (like Mistral-7B), FastAPI, and PostgreSQL to generate intelligent, exciting trip suggestions using a Retrieval-Augmented Generation (RAG) architecture.

# 🧭 Adventura RAGbot

**Adventura RAGbot** is a smart, context-aware chatbot designed to recommend adventure and travel activities across Lebanon. It uses a combination of **Large Language Models (LLMs)**, **FastAPI**, **PostgreSQL**, and **RAG pipelines** to generate personalized and real-time suggestions for users.

---

## 🚀 Features

- ✅ LLM-powered agent using Mistral-7B for intelligent query understanding
- ✅ RAG-based activity suggestions from a live PostgreSQL backend
- ✅ Natural language context classification via LLM (no keyword matching)
- ✅ FastAPI backend with public URL using ngrok
- ✅ Easily embeddable into mobile apps, chatbots, or frontends

---

## 🧠 Tech Stack

- 🧠 [HuggingFace Transformers](https://huggingface.co) – Mistral-7B / LLaMA 2 models
- ⚡ [FastAPI](https://fastapi.tiangolo.com/) – High-performance async web framework
- 🌐 [ngrok](https://ngrok.com/) – Public URL tunneling for local server
- 🗃️ PostgreSQL (or SQLite) – Backend data source for RAG retrieval
- 🔍 RAG Prompt Engineering – Custom prompts + structured activities

---

## 📦 How to Run

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set your HuggingFace and Ngrok tokens in a `.env` file (or directly in `main.py`):
   ```env
   HUGGINGFACE_TOKEN=your_token_here
   NGROK_TOKEN=your_token_here
   ```
4. Run the app:
   ```bash
   python main.py
   ```
5. You’ll see a public URL printed by ngrok. Use `/chat` endpoint like this:
   ```json
   {
     "query": "i want a picnic in ehden",
     "category": 2,
     "location": "Ehden"
   }
   ```

---

## 📁 Folder Structure

```
Adventura-RAGbot/
├── main.py                # Main FastAPI + RAG logic
├── requirements.txt       # Required Python packages
├── README.md              # Project description
├── .env.example           # Token placeholders
├── activities_schema.sql  # DB schema (optional)
```

---

## 📤 Example API Request

```bash
POST /chat
Content-Type: application/json
{
  "query": "looking for a hike near tripoli",
  "category": 1,
  "location": "Tripoli"
}
```

---

## 🤝 Contributions

Got an idea or bug? Feel free to open an issue or submit a pull request.

- Add multilingual support (Arabic, French)
- Add trip scoring and itinerary building
- Support Firebase/Supabase backends

---

## 🔗 Links

- 🤖 [HuggingFace Mistral-7B](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2)
- 📘 [FastAPI Docs](https://fastapi.tiangolo.com/)
- 🛰️ [Ngrok](https://ngrok.com/)
- 💬 [Adel Allam](https://www.linkedin.com/in/adel-allam-4a7378285/) – Project Creator

---

> Made with ❤️ in Lebanon – Let’s plan your next adventure!
