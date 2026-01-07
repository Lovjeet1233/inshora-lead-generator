# Inshora Lead Generator: Unified Insurance AI Agent

Inshora Lead Generator is a sophisticated, multi-channel AI agent system designed for the insurance industry. It integrates telephony, SMS, email, and web-based chat into a unified platform, leveraging state-of-the-art LLMs and RAG (Retrieval-Augmented Generation) to automate lead generation, customer support, and policy management.

## 🚀 Core Features

- **Multi-Channel Communication**: Seamlessly interact with customers via Voice (Telephony), SMS (Twilio), Email, and Web Chat.
- **Advanced RAG System**: Utilizes FAISS for efficient vector search across insurance knowledge bases, including PDFs, websites, and Excel data.
- **CRM Integration**: Built-in support for industry-standard CRMs like **AMS360** and **AgencyZoom**.
- **Real-time Telephony**: Powered by LiveKit and OpenAI Realtime API for low-latency, natural voice conversations.
- **Automated Lead Collection**: Intelligent data collection tools for Home, Auto, Flood, Life, and Commercial insurance.
- **Intelligent Triage**: Automatically escalates complex issues or high-value leads to human agents.

## 🛠 System Architecture & Flow

The system operates through a unified API built with FastAPI, coordinating several specialized services:

### 1. Inbound/Outbound Voice Flow
- **Entry Point**: `agent.py` (LiveKit Worker)
- **Processing**: 
    - **STT**: Deepgram/OpenAI Realtime
    - **LLM**: OpenAI GPT-4o-mini / Realtime Model
    - **TTS**: Cartesia/ElevenLabs
- **Logic**: The agent uses `TelephonyAgent` to handle calls, utilizing tools to query CRMs or collect lead data.

### 2. Chatbot & RAG Flow
- **Entry Point**: `app.py` (`/chat` endpoint)
- **Processing**: 
    - Queries `RAGService` for context from the knowledge base.
    - Maintains conversation state using `thread_id`.
- **Logic**: Combines system instructions with RAG context to provide accurate insurance advice.

### 3. SMS & Email Flow
- **Entry Point**: `routers/sms.py` and `routers/email.py`
- **Processing**: 
    - **SMS**: Twilio API for sending and tracking messages.
    - **Email**: Integrated email service for automated follow-ups.

## 📂 Project Structure

```text
inshora-lead-generator/
├── app.py              # Unified FastAPI Application
├── agent.py            # LiveKit Telephony Agent
├── RAGService.py       # FAISS-based RAG Implementation
├── config/             # Configuration & Prompts
├── services/           # CRM & Insurance Integrations (AMS360, AgencyZoom)
├── routers/            # API Routes (SMS, Email)
├── models/             # Pydantic Data Models
├── tools/              # Agent Function Tools
└── utils/              # Logging & Shared Utilities
```

## ⚙️ Enterprise Readiness Assessment

| Feature | Status | Notes |
| :--- | :--- | :--- |
| **Scalability** | ✅ High | Built on FastAPI and LiveKit; supports async operations. |
| **Integrations** | ✅ Enterprise | Direct SOAP/REST integrations with AMS360 and AgencyZoom. |
| **Data Handling** | ⚠️ Moderate | Currently uses in-memory storage for threads; needs Redis/PostgreSQL for production. |
| **Security** | ⚠️ Moderate | Environment-based secrets; requires OAuth/JWT for API hardening. |
| **RAG Performance** | ✅ High | FAISS provides fast local vector search; easily migratable to Qdrant/Pinecone. |

## 🔧 Setup & Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/AmarBackInField/inshora-lead-generator.git
   cd inshora-lead-generator
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Environment**:
   Create a `.env` file with the following keys:
   - `OPENAI_API_KEY`
   - `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET`
   - `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`, `TWILIO_NUMBER`
   - `AMS360_USERNAME`, `AMS360_PASSWORD` (if using CRM)

4. **Run the Services**:
   - **API**: `uvicorn app:app --reload`
   - **Telephony Agent**: `python agent.py dev`

## 📄 License
[Specify License]

## Credits

Built with:
- [LiveKit](https://livekit.io/) - Real-time communication platform
- [OpenAI](https://openai.com/) - LLM and embeddings
- [FAISS](https://github.com/facebookresearch/faiss) - Vector database
- [Deepgram](https://deepgram.com/) - Speech-to-text
- [Cartesia](https://cartesia.ai/) - Text-to-speech

