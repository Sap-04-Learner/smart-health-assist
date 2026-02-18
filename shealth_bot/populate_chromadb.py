"""
Populate ChromaDB with documents from various file formats.
Supports: TXT, PDF, DOCX, CSV, XLSX
"""

import os
import torch
from pathlib import Path
from tqdm import tqdm
from app.core.rag_pipeline import RAGPipeline
from app.file_parsers import (
    process_directory,
    get_parser,
    get_supported_extensions
)
import gc


# =================== Helper Functions ===================

def _prepare_device(use_gpu: bool) -> str:
    """
    Prepare and validate device for embedding generation.
    
    Args:
        use_gpu: Whether to use GPU if available
        
    Returns:
        Device string: 'cuda' or 'cpu'
    """
    if not use_gpu:
        print('ℹ️  Using CPU for embeddings')
        return 'cpu'
    if not torch.cuda.is_available():
        print("⚠️  GPU requested but not available, falling back to CPU")
        return 'cpu'
    
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"✅ Using GPU: {gpu_name} ({gpu_memory:.1f} GB)")
    return 'cuda'


def _cleanup_device(device: str) -> None:
    """
    Clean up GPU memory after embedding generation.
    
    Args:
        device: Device used ('cuda' or 'cpu')
    """
    if device == 'cuda':
        print("🧹 Cleaning up GPU memory...")
        gc.collect()
        torch.cuda.empty_cache()
        print("✅ GPU memory cleared")


def _create_metadata(
    file_path: str,
    file_name: str | None = None,
    custom_fields: dict | None = None
) -> dict:
    """
    Create standardized metadata for document.
    
    Args:
        file_path: Path to the source file
        file_name: Optional file name (derived from path if not provided)
        custom_fields: Optional additional metadata fields
        
    Returns:
        Dictionary with standardized metadata
    """
    file_name = file_name or Path(file_path).name
    metadata = {
        "source": file_name,
        "file_path": str(file_path),
        "file_type": Path(file_path).suffix,
    }
    
    if custom_fields:
        metadata.update(custom_fields)
    return metadata


# =================== Population Functions ===================

def populate_from_directory(directory_path: str, recursive: bool = True, use_gpu: bool = True):
    """
    Populate ChromaDB from all supported files in a directory.
    
    Args:
        directory_path: Path to directory containing documents
        recursive: Whether to search subdirectories
        use_gpu: Whether to use GPU for embedding generation (faster for bulk operations)
    """
    print(f"\n{'='*60}")
    print(f"📁 Processing files from: {directory_path}")
    print(f"🔍 Recursive search: {recursive}")
    print(f"🧾 Supported formats: {', '.join(get_supported_extensions())}")
    print(f"{'='*60}\n")
    
    # Process all files
    results = process_directory(directory_path, recursive=recursive)
    
    if not results:
        print("⚠️ No supported files found!")
        return
    
    # Prepare device
    device = _prepare_device(use_gpu)
    print(f"🚀 Initializing RAG pipeline with device: {device.upper()}")
    rag = RAGPipeline(device=device)
    
    # Prepare documents
    all_documents, all_metadatas, all_ids = [], [], []
    doc_id = 0

    for file_path, texts in tqdm(results.items(), desc="🧩 Extracting text from files"):
        for text in texts:
            all_documents.append(text)
            all_metadatas.append(_create_metadata(file_path))
            all_ids.append(f"doc_{doc_id}")
            doc_id += 1

    # Batch add to ChromaDB
    if all_documents:
        print(f"\n{'='*60}")
        print(f"💾 Preparing to add {len(all_documents)} documents to ChromaDB...")
        BATCH_SIZE = int(os.getenv('CHROMA_BATCH_SIZE', 5000))
        
        for i in tqdm(range(0, len(all_documents), BATCH_SIZE), desc="Populating ChromaDB", unit="batch"):
            batch_docs = all_documents[i:i + BATCH_SIZE]
            batch_meta = all_metadatas[i:i + BATCH_SIZE]
            batch_ids = all_ids[i:i + BATCH_SIZE]
            
            rag.add_documents(
                documents=batch_docs,
                metadatas=batch_meta,
                ids=batch_ids
            )
        
        print("✅ Successfully populated ChromaDB!")
        print(f"{'='*60}\n")
        # Clean up GPU memory
        _cleanup_device(device)
    else:
        print("⚠️  No text extracted from files!")


def populate_from_file(file_path: str, use_gpu: bool = True):
    """
    Populate ChromaDB from a single file.
    
    Args:
        file_path: Path to the file
        use_gpu: Whether to use GPU for embedding generation
    """
    print(f"\n{'='*60}")
    print(f"📄 Processing single file: {file_path}")
    print(f"{'='*60}\n")
    
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return
    
    try:
        # Get appropriate parser and extract text
        parser = get_parser(file_path)
        texts = parser(file_path)
        
        # Prepare device
        device = _prepare_device(use_gpu)
        print(f"🚀 Initializing RAG pipeline with device: {device.upper()}")
        rag = RAGPipeline(device=device)
        
        # Prepare metadata
        file_name = Path(file_path).name
        metadatas = [_create_metadata(file_path, file_name)] * len(texts)
        ids = [f"doc_{i}" for i in range(len(texts))]
        
        # Add to ChromaDB
        print(f"\n💾 Adding {len(texts)} documents to ChromaDB...")
        rag.add_documents(
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
        print(f"✅ Successfully added documents from {file_name}!")
        print(f"{'='*60}\n")
        
        # Clean up GPU memory
        _cleanup_device(device)
    except Exception as e:
        print(f"❌ Error processing file: {e}")


def populate_from_manual_list(use_gpu: bool = False):
    """
    Populate ChromaDB with manually defined health documents.
    Useful for quick testing or baseline knowledge.
    
    Args:
        use_gpu: Whether to use GPU (usually not needed for small manual lists)
    """
    print(f"\n{'='*60}")
    print("📝 Adding manual health documents...")
    print(f"{'='*60}\n")
    
    health_documents = [
        "Common cold symptoms include runny nose, sore throat, cough, and mild fever. Treatment involves rest, fluids, and over-the-counter medications.",
        "Hypertension is high blood pressure that can lead to heart disease. Management includes lifestyle changes and medications prescribed by a doctor.",
        "Diabetes is characterized by high blood sugar levels. Type 1 requires insulin therapy, while Type 2 can be managed with diet, exercise, and medications.",
        "Migraine headaches cause severe throbbing pain, often with nausea and sensitivity to light. Treatment includes pain relievers and preventive medications.",
        "Asthma causes breathing difficulties due to airway inflammation. Inhalers and avoiding triggers are key management strategies.",
        "Depression is a mental health condition causing persistent sadness and loss of interest. Treatment includes therapy, medication, and lifestyle changes.",
        "Anxiety disorders involve excessive worry and fear. Treatment includes cognitive behavioral therapy, medication, and relaxation techniques.",
        "High cholesterol increases heart disease risk. Management includes diet changes, exercise, and statin medications if needed.",
        "Arthritis causes joint pain and inflammation. Treatment includes pain relief, physical therapy, and anti-inflammatory medications.",
        "Insomnia is difficulty falling or staying asleep. Treatment includes sleep hygiene improvements, cognitive therapy, and sometimes medication."
    ]
    
    device = _prepare_device(use_gpu)
    print(f"🚀 Initializing RAG pipeline with device: {device.upper()}")
    rag = RAGPipeline(device=device)
    
    metadatas = [{"source": "manual_entry", "category": "health_basics"}] * len(health_documents)
    
    rag.add_documents(
        documents=health_documents,
        metadatas=metadatas
    )
    
    print(f"✅ Added {len(health_documents)} manual health documents!")
    print(f"{'='*60}\n")
    _cleanup_device(device)


if __name__ == "__main__":
    USE_GPU_FOR_POPULATION = True
    
    print("\n" + "="*60)
    print("🏥 Smart Health Assist - ChromaDB Population Script")
    print("="*60)
    
    if USE_GPU_FOR_POPULATION:
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"✅ GPU Available: {gpu_name} ({gpu_memory:.1f} GB)")
        else:
            print("⚠️  GPU not available, will use CPU")
    else:
        print("ℹ️  CPU mode selected")
    
    print("="*60 + "\n")
    
    # Option 1: Add manual baseline documents
    # populate_from_manual_list(use_gpu=False)
    
    # Option 2: Process entire directory
    populate_from_directory("./data/documents", recursive=True, use_gpu=USE_GPU_FOR_POPULATION)
    
    # Option 3: Process single file
    # populate_from_file("./data/documents/medical_guide.pdf", use_gpu=USE_GPU_FOR_POPULATION)
    
    print("\n" + "="*60)
    print("✅ Population complete! ChromaDB is ready for queries.")
    print("="*60 + "\n")