"""
rag_engine.py
-------------
Core RAG logic, kept separate from any UI framework so it can be reused
later (e.g. wrapped in a FastAPI endpoint for the company website)
without touching Streamlit code.

Usage:
    from rag_engine import RAGEngine
    engine = RAGEngine()
    answer, sources = engine.ask("What is your refund policy?")
"""

import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Load .env from the same folder as this file, regardless of where the
# script/Streamlit is actually run from. This fixes cases where .env
# exists but isn't picked up because of the current working directory.
load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")

# gemini-embedding-001 replaced the deprecated text-embedding-004 /
# embedding-001 models (Google retired those). Must match ingest.py exactly,
# including EMBEDDING_DIM below — otherwise queries and the stored vectors
# will not be comparable and retrieval will silently return junk (or error).
EMBEDDING_MODEL = "models/gemini-embedding-001"
EMBEDDING_DIM = 3072

MAX_ANSWER_TOKENS = 400  # cap response length (also saves quota)

# Response length presets (word-target used in the prompt; token cap enforced separately)
LENGTH_PRESETS = {
    "brief": "Answer in 1-2 short sentences. Be direct, no extra explanation.",
    "medium": "Answer in 3-5 sentences, or a short bullet list if the source uses bullets.",
    "detailed": "Answer fully and clearly, using bullet points/steps if the source does. Do not omit relevant detail.",
}

# Persona framing: gives the model a consistent identity/tone.
PERSONA = (
    "You are Assist, the official virtual support assistant for this company's "
    "ERP system. You are professional, clear, and friendly — like a knowledgeable "
    "support agent who wants the user to succeed quickly."
)

# Two modes:
#  - "strict": FAQ-only, refuses anything outside the provided context (safer,
#    for production/web).
#  - "general": current testing-phase mode. If the FAQ context doesn't cover
#    the question, Assist may use its own general knowledge to still help, but
#    must clearly flag that the answer isn't from the company's official FAQ.
#    No hard guardrails yet — those get added before the web launch.
SYSTEM_PROMPT_TEMPLATE = """{persona}

You will be given retrieved context from the company's FAQ documents, followed
by the user's question.

{refusal_instructions}

{length_instruction}

Context:
{context}
"""

REFUSAL_STRICT = (
    "Answer the user's question using ONLY the context above. If the context "
    "does not contain the answer, say clearly that you don't have that "
    "information in the FAQ documents and suggest the user contact support. "
    "Do not use outside knowledge and do not make anything up."
)

REFUSAL_GENERAL = (
    "First try to answer using ONLY the context above, since it is the "
    "company's official, verified source. If — and only if — the context does "
    "not cover the question, you may answer using your own general knowledge "
    "instead. In that case, you MUST start the answer with the exact tag "
    "'[General knowledge - not from official FAQ]' before the rest of your "
    "reply, so the user knows this wasn't verified against company documents. "
    "This is a testing-phase behavior; do not use it to override or contradict "
    "anything the context does say."
)


class RAGEngine:
    def __init__(self, k: int = 4):
        pinecone_api_key = os.getenv("PINECONE_API_KEY", "").strip()
        pinecone_index_name = os.getenv("PINECONE_INDEX_NAME", "erp-faq-index").strip()

        if not pinecone_api_key:
            raise EnvironmentError(
                "PINECONE_API_KEY not set. Add it to your .env file (local) "
                "or your Render Environment tab (production)."
            )

        gemini_key_for_embeddings = os.getenv("GEMINI_API_KEY", "").strip()
        if not gemini_key_for_embeddings:
            raise EnvironmentError(
                "GEMINI_API_KEY not set. It's required for embeddings even if "
                "you're using Groq/OpenRouter as your chat LLM, since queries "
                "must be embedded with the same model used in ingest.py."
            )

        embeddings = GoogleGenerativeAIEmbeddings(
            model=EMBEDDING_MODEL,
            google_api_key=gemini_key_for_embeddings,
            output_dimensionality=EMBEDDING_DIM,
        )

        # Connects to the existing Pinecone index built by ingest.py.
        # Does NOT create or upload anything here — this is read-only access
        # at query time.
        self.vectorstore = PineconeVectorStore(
            index_name=pinecone_index_name,
            embedding=embeddings,
            pinecone_api_key=pinecone_api_key,
        )
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})

        self.llms = self._build_llm_chain()
        if not self.llms:
            raise EnvironmentError(
                "No valid LLM API keys found. Set at least one of "
                "GROQ_API_KEY, GEMINI_API_KEY, or OPENROUTER_API_KEY in your .env file."
            )

        # Prompt is built per-call in ask(), since it depends on the chosen
        # response length and strict/general mode.

    def _build_llm_chain(self):
        """
        Builds an ordered list of (name, llm) pairs to try in sequence:
        Groq -> Gemini -> OpenRouter. Only providers with a real (non-
        placeholder) API key set in .env are included.
        """
        candidates = []

        groq_key = os.getenv("GROQ_API_KEY", "").strip()
        if groq_key and groq_key != "your_free_groq_api_key_here":
            candidates.append((
                "Groq",
                ChatGroq(
                    model="openai/gpt-oss-120b",
                    temperature=0.2,
                    max_tokens=MAX_ANSWER_TOKENS,
                    api_key=groq_key,
                ),
            ))

        gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
        if gemini_key and gemini_key != "your_free_gemini_api_key_here":
            candidates.append((
                "Gemini",
                ChatGoogleGenerativeAI(
                    model="gemini-1.5-flash",
                    temperature=0.2,
                    max_output_tokens=MAX_ANSWER_TOKENS,
                    google_api_key=gemini_key,
                ),
            ))

        openrouter_key = os.getenv("OPENROUTER_API_KEY", "").strip()
        if openrouter_key and openrouter_key != "your_free_openrouter_api_key_here":
            candidates.append((
                "OpenRouter",
                ChatOpenAI(
                    model="meta-llama/llama-3.1-8b-instruct:free",
                    temperature=0.2,
                    max_tokens=MAX_ANSWER_TOKENS,
                    api_key=openrouter_key,
                    base_url="https://openrouter.ai/api/v1",
                ),
            ))

        return candidates

    @staticmethod
    def _format_docs(docs) -> str:
        return "\n\n---\n\n".join(d.page_content for d in docs)

    def ask(self, question: str, length: str = "medium", mode: str = "general"):
        """
        Returns (answer: str, sources: list[dict]).

        length: "brief" | "medium" | "detailed" — controls how long the answer is.
        mode:   "strict"  -> FAQ-only, refuses anything outside the context.
                "general" -> testing-phase mode: falls back to the model's own
                             general knowledge (clearly flagged) if the FAQ
                             context doesn't cover the question.

        Tries each configured LLM provider in order (Groq -> Gemini ->
        OpenRouter); if one fails (rate limit, invalid key, downtime, etc.)
        it automatically falls through to the next one instead of crashing.
        """
        length_instruction = LENGTH_PRESETS.get(length, LENGTH_PRESETS["medium"])
        refusal_instructions = REFUSAL_STRICT if mode == "strict" else REFUSAL_GENERAL

        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
            persona=PERSONA,
            refusal_instructions=refusal_instructions,
            length_instruction=length_instruction,
            context="{context}",
        )
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{question}"),
            ]
        )

        try:
            retrieved_docs = self.retriever.invoke(question)
        except Exception as e:
            raise RuntimeError(
                f"Failed to retrieve context from Pinecone: {e}. "
                "Check that PINECONE_INDEX_NAME matches an existing index, "
                "and that its dimension matches EMBEDDING_DIM in this file."
            ) from e

        context = self._format_docs(retrieved_docs)

        last_error = None
        answer = None
        for name, llm in self.llms:
            try:
                chain = prompt | llm | StrOutputParser()
                answer = chain.invoke({"context": context, "question": question})
                break
            except Exception as e:
                last_error = e
                print(f"[RAGEngine] {name} failed ({e}), trying next provider...")
                continue

        if answer is None:
            raise RuntimeError(
                f"All configured LLM providers failed. Last error: {last_error}"
            )

        # Hard cap as a final safety net in case a provider ignores
        # max_tokens/max_output_tokens.
        if len(answer) > MAX_ANSWER_TOKENS * 6:  # rough chars-per-token guard
            answer = answer[: MAX_ANSWER_TOKENS * 6].rsplit(" ", 1)[0] + "..."

        sources = [
            {
                "source": doc.metadata.get("source", "unknown"),
                "snippet": doc.page_content[:200],
            }
            for doc in retrieved_docs
        ]
        return answer, sources