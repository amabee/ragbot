import os
import argparse
from pathlib import Path
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import (
    TextLoader,
    PDFPlumberLoader,
    UnstructuredMarkdownLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
import spacy

# Paths
PER_BOOK_PATH = Path("chroma_per_book")
BOOKS_INDEX_PATH = Path("books_index")

# Load spaCy for NER
nlp = spacy.load("en_core_web_sm")

# Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-large-en-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)


def extract_characters(text):
    """Extract lowercase unique character/entity names from text."""
    doc = nlp(text)
    characters = {
        ent.text.lower() for ent in doc.ents if ent.label_ in {"PERSON", "ORG", "GPE"}
    }
    return sorted(characters)


def clean_metadata(metadata):
    """
    Ensure all metadata values are valid for Chroma (str, int, float, bool, None).
    Lists are converted to comma-separated strings for storage.
    """
    cleaned = {}
    for key, value in metadata.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            cleaned[key] = value
        elif isinstance(value, list):
            cleaned[key] = ", ".join(map(str, value))  # store as string for Chroma
        else:
            cleaned[key] = str(value)
    return cleaned


def load_document(file_path):
    """Load document based on extension."""
    ext = file_path.suffix.lower()
    if ext == ".txt":
        return TextLoader(str(file_path)).load()
    elif ext == ".pdf":
        return PDFPlumberLoader(str(file_path)).load()
    elif ext in (".md", ".markdown"):
        return UnstructuredMarkdownLoader(str(file_path)).load()
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def process_file(file_path):
    print(f"📚 Processing file: {file_path}")
    docs = load_document(file_path)

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    print(f"  ✂️ Split into {len(chunks)} chunks.")

    for chunk in chunks:
        # Detect characters/entities
        characters = extract_characters(chunk.page_content)

        # Keep original lowercase list for later use
        chunk.metadata["characters"] = characters

        # Add clean metadata for Chroma
        chunk.metadata["book_name"] = file_path.stem.lower()
        chunk.metadata = clean_metadata(chunk.metadata)

    return chunks


def build_per_book_db(file_path):
    """Build Chroma DB for a single book."""
    chunks = process_file(file_path)
    book_name = file_path.stem.lower()
    persist_dir = PER_BOOK_PATH / book_name
    persist_dir.mkdir(parents=True, exist_ok=True)
    db = Chroma.from_documents(chunks, embeddings, persist_directory=str(persist_dir))
    print(f"  ✅ Saved {len(chunks)} chunks to {persist_dir}")


def build_books_index(all_books):
    """Create global index for all books."""
    from langchain.schema import Document

    docs = []
    for book in all_books:
        docs.append(
            Document(
                page_content=f"Index entry for {book.stem}",
                metadata={"book_name": book.stem.lower()},
            )
        )
    db = Chroma.from_documents(
        docs, embeddings, persist_directory=str(BOOKS_INDEX_PATH)
    )
    print(f"📖 Global book index created with {len(docs)} entries.")


def main():
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=str, help="Directory with story files.")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    files = [
        f
        for f in data_dir.iterdir()
        if f.suffix.lower() in (".txt", ".pdf", ".md", ".markdown")
    ]

    if not files:
        print("❌ No supported files found.")
        return

    PER_BOOK_PATH.mkdir(exist_ok=True)
    for file_path in files:
        try:
            build_per_book_db(file_path)
        except Exception as e:
            print(f"  ❌ Error processing {file_path}: {e}")

    build_books_index(files)


if __name__ == "__main__":
    main()
