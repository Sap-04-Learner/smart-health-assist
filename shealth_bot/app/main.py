from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from contextlib import asynccontextmanager
import logging
import json

# === FIX: Use absolute imports ===
from .config.setting import get_settings, ensure_directories
from .config.logging import setup_logging
from .core.rag_pipeline import RAGPipeline
from .core.session_manager import SessionManager
from .schemas.models import ChatRequest, AddDocumentsRequest  # or from schemas.models if moved

logger = logging.getLogger(__name__)

settings = get_settings()

# Initialize core components (after settings)
rag_pipeline = RAGPipeline()  # device from settings.runtime by default
session_manager = SessionManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    ensure_directories(settings)
    setup_logging(settings)
    
    logger.info("Starting Health Assistant RAG Server...")
    logger.info(f"   Ollama Model: {settings.ollama_model}")
    logger.info(f"   ChromaDB Path: {settings.chroma_db_path}")
    logger.info(f"   Embedding Device: {rag_pipeline.device.upper()}")
    logger.info(f"   Documents in collection: {rag_pipeline.collection.count()}")
    
    yield
    
    logger.info("Shutting down Health Assistant RAG Server...")


app = FastAPI(
    title='Health Assistant RAG API',
    description='RAG-powered conversational chatbot for health queries',
    version=settings.app_version,
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=settings.allowed_methods,
    allow_headers=settings.allowed_headers,
)


# =================== Exception Handlers ===================

@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    logger.error(f"Validation error: {exc}")
    raise HTTPException(status_code=400, detail=str(exc))


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unexpected error: {exc}", exc_info=True)
    raise HTTPException(status_code=500, detail="Internal server error")


# =================== Health Check ===================

@app.get('/')
async def root():
    return {
        'status': 'running',
        'model': settings.ollama_model,
        'documents': rag_pipeline.collection.count(),
        'embedding_device': rag_pipeline.device,
        'version': settings.app_version
    }


# =================== STREAMING CHAT ENDPOINT ===================

@app.post('/chat')
async def chat(request: ChatRequest):
    """
    Streaming chat endpoint — returns text/event-stream
    """
    logger.info(f"Chat stream request | user: {request.user_id} | session: {request.session_id}")

    try:
        # 1. Get history
        history = session_manager.get_history(
            request.user_id,
            request.session_id,
            max_messages=50  # more than request.max_history for context
        )

        # 2. Retrieve context
        retrieval_results = rag_pipeline.retrieve_context(request.message, n_results=3)
        context_docs = retrieval_results['documents'][0] if retrieval_results['documents'] else []
        sources = rag_pipeline.format_sources(retrieval_results)

        # 3. Save user message immediately
        session_manager.save_message(request.user_id, request.session_id, 'user', request.message)

        # 4. Streaming generator
        async def stream_response():
            full_response = ""
            try:
                stream_gen = rag_pipeline.generate_response(
                    query=request.message,
                    context_docs=context_docs,
                    history=history,
                    sources=sources,
                    stream=True
                )

                async for chunk in stream_gen:
                    if chunk:
                        full_response += chunk
                        # Send chunk as Server-Sent Event
                        yield f"data: {json.dumps({'content': chunk})}\n\n"

                # End of stream
                yield f"data: {json.dumps({'done': True, 'sources': sources})}\n\n"

                # Save assistant response after complete
                session_manager.save_message(
                    request.user_id,
                    request.session_id,
                    'assistant',
                    full_response
                )

            except Exception as e:
                logger.error(f"Streaming error: {e}")
                error_msg = "Sorry, I couldn't generate a response."
                yield f"data: {json.dumps({'error': error_msg})}\n\n"
                session_manager.save_message(request.user_id, request.session_id, 'assistant', error_msg)

        return StreamingResponse(
            stream_response(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",  # Disable nginx buffering if behind proxy
            }
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Chat setup error: {e}")
        raise HTTPException(status_code=500, detail="Failed to process request")


# =================== Other Endpoints (unchanged but cleaned) ===================

@app.post('/add-documents')
async def add_documents(request: AddDocumentsRequest):
    if not request.documents:
        return JSONResponse(
            status_code=400,
            content={"status": "error", "message": "No documents provided", "count": 0}
        )

    # Use validator from models.py (now automatic)
    rag_pipeline.add_documents(request.documents)
    
    return {
        'status': 'success',
        'count': len(request.documents),
        'total_documents': rag_pipeline.collection.count()
    }


@app.get('/collection-info')
async def collection_info():
    return {
        'collection_name': settings.collection_name,
        'document_count': rag_pipeline.collection.count(),
        'embedding_device': rag_pipeline.device,
        'embedding_model': settings.embedding_model
    }


@app.get('/sessions/{user_id}')
async def list_sessions(user_id: str):
    sessions = session_manager.list_sessions(user_id)
    return {'user_id': user_id, 'sessions': sessions, 'count': len(sessions)}


@app.delete('/sessions/{user_id}/{session_id}')
async def clear_session(user_id: str, session_id: str):
    session_manager.clear_session(user_id, session_id)
    return {'status': 'success', 'message': f'Session {session_id} cleared'}


# =================== Run Server ===================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",  # assuming file is main.py in project root
        host=settings.server_host,
        port=settings.server_port,
        reload=True,
        log_level="info"
    )