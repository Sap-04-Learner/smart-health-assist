from pydantic import BaseModel, Field, field_validator


class ChatRequest(BaseModel):
    """Request model for chat endpoint with comprehensive validation."""
    
    user_id: str = Field(
        ...,
        min_length=1,
        max_length=128,
        pattern=r'^[a-zA-Z0-9_-]+$',
        description='Unique User Identifier (alphanumeric, underscore, dash only)'
    )
    
    session_id: str = Field(
        ...,
        min_length=1,
        max_length=64,
        pattern=r'^[a-zA-Z0-9_-]+$',
        description='Session Identifier (alphanumeric, underscore, dash only)'
    )
    
    message: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="User message (1-2000 characters)"
    )
    
    max_history: int = Field(
        default=20,
        ge=1,
        le=50,
        description="Number of previous messages to include in context (1-50)"
    )


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""
    
    response: str = Field(..., description="Assistant's response")
    sources: list[str] | None = Field(
        default=None,
        description="Medical sources referenced (e.g., document titles or URLs)"
    )
    session_id: str = Field(..., description="Session identifier")


class AddDocumentsRequest(BaseModel):
    """Request model for adding documents with validation."""
    
    documents: list[str] = Field(
        ...,
        min_items=1,
        max_items=500,  # Reduced from 1000 → safer default
        description="List of documents to add (1-500 items)"
    )
    
    @field_validator('documents')
    @classmethod
    def validate_documents(cls, docs: list[str]) -> list[str]:
        """Enforce quality criteria per document."""
        for i, doc in enumerate(docs):
            stripped = doc.strip()
            if len(stripped) < 10:
                raise ValueError(f"Document {i} too short (min 10 chars after strip)")
            if len(stripped) > 10000:
                raise ValueError(f"Document {i} too long (max 10000 chars)")
            if len(stripped.split()) < 3:
                raise ValueError(f"Document {i} must have at least 3 words")
        return docs


class CollectionInfoResponse(BaseModel):
    """Response model for collection info endpoint."""
    
    collection_name: str = Field(..., description="ChromaDB collection name")
    document_count: int = Field(..., ge=0, description="Number of documents in collection")
    embedding_device: str = Field(..., description="Device used for embeddings")
    embedding_model: str = Field(..., description="Embedding model name")