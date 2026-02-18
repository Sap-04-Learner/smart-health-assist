# rag_pipeline.py
import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer
import torch
import ollama
from typing import Any, Literal, AsyncGenerator
import logging

# Use absolute import from project root structure
from ..config.setting import get_settings

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    RAG (Retrieval-Augmented Generation) pipeline for health queries.
    Integrates ChromaDB, embedding models, and Ollama LLM.
    """
    VALID_DEVICES = {'cpu', 'cuda', 'auto'}

    def __init__(self, device: str | None = None):
        """
        Initialize RAG Pipeline with device validation.
        """
        settings = get_settings()

        if device is None:
            device = settings.embedding_device_runtime
        elif device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        if device not in self.VALID_DEVICES:
            raise ValueError(
                f"Invalid device '{device}'. Must be one of: {self.VALID_DEVICES}"
            )

        self.device = device
        logger.info(f"Using device: {self.device.upper()} for embeddings")

        # Initialize ChromaDB client
        self.chroma_client = chromadb.PersistentClient(
            path=settings.chroma_db_path,
            settings=ChromaSettings()
        )

        # Initialize embedding model
        self.embedding_model = SentenceTransformer(
            settings.embedding_model,
            device=self.device
        )

        # Get or create collection
        self.collection = self.chroma_client.get_or_create_collection(
            name=settings.collection_name,
            metadata={"hnsw:space": "cosine"}
        )

        logger.info(f"ChromaDB initialized with {self.collection.count()} documents")

    def add_documents(
        self,
        documents: list[str],
        metadatas: list[dict[str, Any]] | None = None,
        ids: list[str] | None = None,
    ) -> None:
        """Add documents to ChromaDB with embeddings."""
        if not documents:
            logger.warning("No documents provided to add.")
            return

        if ids is None:
            ids = [f"doc_{i}" for i in range(len(documents))]

        logger.info(f"Generating embeddings for {len(documents)} documents on {self.device.upper()}...")

        embeddings = self.embedding_model.encode(
            documents,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        ).tolist()

        self.collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )

        logger.info(f"Successfully added {len(documents)} documents to ChromaDB")

    def retrieve_context(
        self,
        query: str,
        n_results: int = 3
    ) -> dict[str, Any]:
        """Retrieve relevant context from ChromaDB."""
        query_embedding = self.embedding_model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        ).tolist()

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )

        return results

    def _user_wants_direct_answer(self, history: list[dict[str, Any]]) -> bool:
        """Detect if user explicitly wants a direct answer (frustration signals)."""
        if len(history) < 3:
            return False

        frustration_phrases = [
            'just answer', 'give me the answer', 'just tell me', 'answer now',
            'stop asking', 'no more questions', 'frustrated', 'just answer man',
            'answer me', 'answer please', 'i need answer', 'provide answer'
        ]

        user_messages = [msg['content'].lower() for msg in history[-5:] if msg['role'] == 'user']

        return any(any(phrase in msg for phrase in frustration_phrases) for msg in user_messages)

    def _extract_conversation_summary(self, old_messages: list[dict[str, Any]]) -> str:
        """Extract key medical information from older messages."""
        conditions = []
        important_info = []

        for msg in old_messages:
            if msg['role'] != 'user':
                continue
            content_lower = msg['content'].lower()

            # Simple keyword-based extraction (can be improved with NER later)
            if 'diabetes' in content_lower or 'diabetic' in content_lower:
                if 'type 2' in content_lower:
                    conditions.append("Type 2 Diabetes")
                elif 'type 1' in content_lower:
                    conditions.append("Type 1 Diabetes")
                else:
                    conditions.append("Diabetes")

            if 'hypertension' in content_lower or 'high blood pressure' in content_lower:
                conditions.append("Hypertension")

            if 'heart disease' in content_lower or 'cardiac' in content_lower:
                conditions.append("Heart disease")

            if 'asthma' in content_lower:
                conditions.append("Asthma")

            # Context flags
            if 'recently' in content_lower or 'just' in content_lower:
                important_info.append("Recently diagnosed")
            if 'doctor' in content_lower:
                important_info.append("Medical appointment scheduled")
            if 'no remedies' in content_lower or 'never tried' in content_lower:
                important_info.append("No current treatment")
            if 'diet' in content_lower:
                important_info.append("Seeking dietary advice")
            if 'precaution' in content_lower or 'prevent' in content_lower:
                important_info.append("Wants preventive measures")

        summary_parts = []
        if conditions:
            summary_parts.append(f"Patient Condition(s): {', '.join(set(conditions))}")
        if important_info:
            summary_parts.append("Key Context: " + "; ".join(set(important_info)))

        return "\n".join(summary_parts) if summary_parts else ""

    def format_sources(self, retrieval_results: dict[str, Any]) -> list[str]:
        """Format sources from retrieval results safely."""
        formatted_sources = []

        if not retrieval_results.get('metadatas'):
            return formatted_sources

        metadatas = retrieval_results['metadatas'][0]

        for metadata in metadatas:
            if not metadata:
                continue

            question = metadata.get('question', 'Unknown question')
            source = metadata.get('source', 'Medical Database')
            focus_area = metadata.get('focus_area', '')

            source_str = f'question: {question} | source: {source}'
            if focus_area:
                source_str += f' | focus_area: {focus_area}'

            formatted_sources.append(source_str)

        return formatted_sources

    def generate_response(
        self,
        query: str,
        context_docs: list[str],
        history: list[dict[Literal["role", "content"], str]],
        sources: list[str],
        stream: bool = False
    ) -> str | AsyncGenerator[str, None]:
        """
        Generate response using Ollama.
        Supports both full string return (non-stream) and async generator (stream).
        """
        settings = get_settings()

        system_prompt = f"""You are Smart Health Assist — an intelligent, empathetic medical assistant.
Use only the provided medical knowledge and conversation history.
When discussing NEW symptoms, ask 2-3 clarifying questions before giving advice.
If the user seems frustrated or explicitly asks for a direct answer, provide one without questions.

Medical Sources Used:
{'\n'.join(sources or ['Internal medical knowledge base'])}

Retrieved Medical Knowledge:
{'\n\n'.join(context_docs)}
"""

        messages = [{"role": "system", "content": system_prompt}]

        if len(history) > 10:
            older_summary = self._extract_conversation_summary(history[:-10])
            if older_summary:
                messages.append({
                    "role": "system",
                    "content": f"Previous relevant medical context summary:\n{older_summary}"
                })

        for msg in history[-10:]:
            messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })

        messages.append({"role": "user", "content": query})

        try:
            if stream:
                async def stream_chunks():
                    stream_response = ollama.chat(
                        model=settings.ollama_model,
                        messages=messages,
                        stream=True
                    )
                    for chunk in stream_response:
                        content = chunk.get('message', {}).get('content', '')
                        if content:
                            yield content

                return stream_chunks()

            else:
                response = ollama.chat(
                    model=settings.ollama_model,
                    messages=messages
                )
                return response['message']['content']

        except Exception as e:
            logger.error(f"Ollama generation failed: {e}", exc_info=True)
            error_text = "Sorry, I couldn't generate a response right now. Please try again later."
            if stream:
                async def error_stream():
                    yield error_text
                return error_stream()
            else:
                return error_text