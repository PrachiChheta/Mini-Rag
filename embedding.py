import json
import numpy as np
import pickle
import os
import re
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import wordpunct_tokenize
from nltk.corpus import stopwords
from pathlib import Path

# Constants for chunking
EXCLUDED_SECTIONS = {
    "abstract", "references", "table of contents", "list of tables", "list of figures"
}
SECTION_PATTERN = re.compile(r"^(#{1,6})\s+(.*)")
TABLE_PATTERN = re.compile(
    r"(Table[^\n]*)\n(?:\n*)"
    r"((?:\|.*\|\n?)+)",
    re.IGNORECASE
)
IMAGE_PATTERN = re.compile(r"!\[.*?\]\(\s*<?(.*?)>?\s*\)")

# Chunking parameters
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200

# Set cache directories
os.environ["HF_HOME"] = "/opt/dlami/nvme/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "/opt/dlami/nvme/hf_cache"
os.environ["HF_DATASETS_CACHE"] = "/opt/dlami/nvme/hf_cache"

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

class MetadataAndEmbeddingCreator:
    """Creates metadata from Markdown files and generates embeddings for RAG pipeline"""
    
    def __init__(self, markdown_directory: str = "outputs", metadata_path: str = "scibert_metadata.json"):
        """
        Initialize the metadata and embedding creator
        
        Args:
            markdown_directory: Base directory containing markdown files (expects outputs/{folder}/{folder}-referenced.md)
            metadata_path: Path to save the metadata JSON file
        """
        self.markdown_directory = markdown_directory
        self.metadata_path = metadata_path
        self.bm25_index_path = "bm25_index.pkl"
        self.bm25_docs_path = "bm25_documents.pkl"
        self.embeddings_path = "document_embeddings.npy"
        self.metadata = []
        
        print("🚀 Initializing Metadata and Embedding Creator...")
        
        # Load SciBERT embedding model
        print("📥 Loading SciBERT embedding model (cased)...")
        try:
            self.embedder = SentenceTransformer("allenai/scibert_scivocab_cased")
            print("✅ SciBERT (cased) model loaded successfully!")
        except Exception as e:
            print(f"❌ Failed to load SciBERT model: {str(e)}")
            print("🔄 Falling back to MiniLM model...")
            self.embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    
    def normalize(self, text):
        """Normalize text for comparison"""
        return text.strip().lower()
    
    def extract_sections(self, md_text):
        """Split markdown text into sections by headings."""
        sections = []
        current = {"title": None, "content": []}
        for line in md_text.splitlines():
            match = SECTION_PATTERN.match(line)
            if match:
                if current["title"]:
                    sections.append(current)
                current = {"title": match.group(2).strip(), "content": []}
            else:
                current["content"].append(line)
        if current["title"]:
            sections.append(current)
        return sections
    
    def extract_tables(self, text):
        """Find markdown tables and convert them to structured format."""
        tables = []
        for match in TABLE_PATTERN.finditer(text):
            caption = match.group(1).strip()
            table_markdown = match.group(2).strip()
            lines = [line.strip() for line in table_markdown.splitlines() if line.strip()]

            if len(lines) < 2:
                continue

            rows = [[cell.strip() for cell in row.strip("|").split("|")] for row in lines]

            headers = rows[0]
            if re.fullmatch(r"[-:\s|]+", lines[1]):
                data_rows = rows[2:]
            else:
                data_rows = rows[1:]

            structured = {
                "caption": caption,
                "headers": headers,
                "rows": data_rows
            }
            tables.append(structured)
        return tables
    
    def extract_images(self, text):
        """Find image paths."""
        return IMAGE_PATTERN.findall(text)
    
    def find_table_positions(self, text):
        """Find start and end positions of tables in text."""
        positions = []
        for match in TABLE_PATTERN.finditer(text):
            positions.append((match.start(), match.end()))
        return positions

    def find_image_positions(self, text):
        """Find start and end positions of images in text."""
        positions = []
        for match in IMAGE_PATTERN.finditer(text):
            positions.append((match.start(), match.end()))
        return positions

    def is_position_in_ranges(self, pos, ranges):
        """Check if position is within any of the given ranges."""
        for start, end in ranges:
            if start <= pos <= end:
                return True
        return False

    def recursive_character_split(self, text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
        """Split text into chunks while preserving complete tables and images."""
        if len(text) <= chunk_size:
            return [text]
        
        table_positions = self.find_table_positions(text)
        image_positions = self.find_image_positions(text)
        protected_ranges = table_positions + image_positions
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            if end >= len(text):
                chunks.append(text[start:])
                break
            
            split_point = end
            
            for i in range(end, max(start + chunk_size // 2, start + 1), -1):
                if self.is_position_in_ranges(i, protected_ranges):
                    continue
                
                if i > 0 and text[i-1:i+1] == '\n\n':
                    split_point = i
                    break
                elif i > 0 and text[i-1] in '.!?' and text[i] in ' \n':
                    split_point = i
                    break
                elif i > 0 and text[i-1] == '\n':
                    split_point = i
                    break
            
            if split_point == end:
                for range_start, range_end in protected_ranges:
                    if range_start <= split_point <= range_end:
                        split_point = range_end + 1
                        break
            
            chunk = text[start:split_point].strip()
            if chunk:
                chunks.append(chunk)
            
            next_start = split_point - overlap
            
            for range_start, range_end in protected_ranges:
                if range_start <= next_start <= range_end:
                    next_start = range_end + 1
                    break
            
            start = max(next_start, split_point)
        
        return chunks

    def remove_tables(self, text):
        """Remove tables and captions from text."""
        return TABLE_PATTERN.sub("", text).strip()

    def fix_image_paths(self, md_path: Path, image_paths):
        """Converts relative image paths to paths relative to project root."""
        PROJECT_ROOT = Path.cwd()
        fixed_paths = []
        for p in image_paths:
            abs_path = (md_path.parent / p).resolve()
            try:
                rel_path = abs_path.relative_to(PROJECT_ROOT)
            except ValueError:
                rel_path = abs_path
            fixed_paths.append(str(rel_path))
        return fixed_paths
    
    def extract_text_from_markdown(self, md_path: Path) -> List[Dict]:
        """
        Extract text from Markdown file and create chunks (using advanced chunking logic)
        
        Args:
            md_path: Path to markdown file
            
        Returns:
            List of chunk dictionaries
        """
        chunks = []
        md_name = md_path.name
        folder_name = md_path.parent.name
        
        try:
            print(f"   📄 Processing: {md_name} from {folder_name}")
            
            with open(md_path, 'r', encoding='utf-8') as f:
                md_text = f.read()
            
            # Extract sections
            sections = self.extract_sections(md_text)
            
            # Filter out excluded sections
            filtered_sections = []
            for section in sections:
                title = section["title"]
                if not title:
                    continue
                norm = self.normalize(title)
                if any(excl in norm for excl in EXCLUDED_SECTIONS):
                    continue
                filtered_sections.append(section)
            
            # Combine all non-excluded sections
            combined_text = ""
            section_boundaries = {}  # Track which sections are in which chunks
            
            for section in filtered_sections:
                section_text = "\n".join(section["content"])
                section_start = len(combined_text)
                combined_text += f"\n\n## {section['title']}\n\n{section_text}"
                section_boundaries[section_start] = section['title']
            
            combined_text = combined_text.strip()
            
            # Split into chunks using advanced chunking
            text_chunks = self.recursive_character_split(combined_text)
            
            # Process each chunk
            for i, chunk_text in enumerate(text_chunks):
                # Find which section this chunk belongs to
                section_title = "Unknown Section"
                for section in filtered_sections:
                    if section['title'] in chunk_text:
                        section_title = section['title']
                        break
                
                # Extract tables and images from this chunk
                tables = self.extract_tables(chunk_text)
                images = self.extract_images(chunk_text)
                images = self.fix_image_paths(md_path, images)
                
                # Remove tables from text (keep structured data)
                clean_text = self.remove_tables(chunk_text)
                
                chunk = {
                    "text": clean_text,
                    "section_title": section_title,
                    "pdf": folder_name,  # Use folder name as document identifier
                    "page": 1,  # Markdown doesn't have pages
                    "tables": tables,
                    "images": images
                }
                chunks.append(chunk)
            
            print(f"      ✅ Extracted {len(chunks)} chunks from {md_name}")
            
        except Exception as e:
            print(f"      ❌ Error processing {md_name}: {e}")
        
        return chunks
    
    def create_metadata_from_markdown(self):
        """
        Create metadata JSON from all markdown files in outputs directory
        Following the pattern: outputs/{folder}/{folder}-referenced.md
        """
        print(f"\n📚 Creating metadata from markdown files in: {self.markdown_directory}")
        
        # Check if directory exists
        if not os.path.exists(self.markdown_directory):
            print(f"❌ ERROR: Directory not found: {self.markdown_directory}")
            print(f"Please create the directory and add markdown files.")
            return False
        
        base_path = Path(self.markdown_directory)
        md_files = []
        
        # Look for folders with markdown files: outputs/{folder}/{folder}-referenced.md
        for folder in base_path.iterdir():
            if not folder.is_dir():
                continue
            md_file = folder / f"{folder.name}-referenced.md"
            if md_file.exists():
                md_files.append(md_file)
                print(f"   Found: {folder.name}/{md_file.name}")
        
        if not md_files:
            print(f"❌ ERROR: No markdown files found in {self.markdown_directory}")
            print(f"   Looking for pattern: {{folder}}/{{folder}}-referenced.md")
            return False
        
        print(f"📊 Found {len(md_files)} markdown files")
        
        # Process each markdown file
        all_chunks = []
        chunk_counter = 1
        
        for md_path in md_files:
            chunks = self.extract_text_from_markdown(md_path)
            
            # Add chunk IDs
            for chunk in chunks:
                chunk["chunk_id"] = f"chunk-{chunk_counter:03d}"
                all_chunks.append(chunk)
                chunk_counter += 1
        
        self.metadata = all_chunks
        
        # Save metadata
        print(f"\n💾 Saving metadata to {self.metadata_path}...")
        with open(self.metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Metadata created with {len(self.metadata)} total chunks")
        return True

    def prepare_text_for_search(self, chunk: Dict) -> str:
        """
        Prepare text content from chunk for search indexing
        (Same as in your RAG code)
        
        Args:
            chunk: Dictionary containing chunk data
        Returns:
            Combined text string for indexing
        """
        text_parts = []
        
        # Add main text
        if chunk.get("text", "").strip():
            text_parts.append(chunk["text"].strip())
        
        # Add section title
        if chunk.get("section_title", "").strip():
            text_parts.append(chunk["section_title"].strip())
        
        # Add table information
        if chunk.get("tables"):
            for table in chunk["tables"]:
                if table.get("caption"):
                    text_parts.append(f"Table: {table['caption']}")
                if table.get("headers"):
                    headers_text = " ".join(table["headers"])
                    text_parts.append(f"Headers: {headers_text}")
        
        # Combine all parts
        combined_text = " ".join(text_parts).strip()
        
        # Fallback if no text content
        if not combined_text:
            combined_text = f"Document chunk from {chunk.get('pdf', 'unknown')}"
        
        return combined_text

    def create_bm25_index(self):
        """Create and save BM25 index for keyword-based search"""
        if not self.metadata:
            print("❌ No metadata available. Create metadata first.")
            return False
            
        print("\n🔨 Building BM25 index...")
        bm25_documents = []
        stop_words = set(stopwords.words('english'))
        
        for i, chunk in enumerate(self.metadata):
            if (i + 1) % 500 == 0:
                print(f"   Processing document {i + 1}/{len(self.metadata)}")
            
            text_content = self.prepare_text_for_search(chunk)
            tokens = wordpunct_tokenize(text_content.lower())

            filtered_tokens = [token for token in tokens 
                             if token.isalnum() and token not in stop_words]
            bm25_documents.append(filtered_tokens)
        
        # Create BM25 index
        bm25 = BM25Okapi(bm25_documents)
        
        # Save BM25 index and documents
        print("💾 Saving BM25 index...")
        with open(self.bm25_index_path, 'wb') as f:
            pickle.dump(bm25, f)
        with open(self.bm25_docs_path, 'wb') as f:
            pickle.dump(bm25_documents, f)
        
        print(f"✅ BM25 index created and saved with {len(bm25_documents)} documents")
        print(f"   - Index saved to: {self.bm25_index_path}")
        print(f"   - Documents saved to: {self.bm25_docs_path}")
        return True

    def create_embeddings(self):
        """Create and save document embeddings for semantic search"""
        if not self.metadata:
            print("❌ No metadata available. Create metadata first.")
            return False
            
        print("\n🔨 Computing document embeddings...")
        print("ℹ️  This may take some time depending on corpus size...")
        
        texts_for_embedding = []
        for chunk in self.metadata:
            text = self.prepare_text_for_search(chunk)
            texts_for_embedding.append(text)
        
        # Compute embeddings in batches
        batch_size = 32
        all_embeddings = []
        total_batches = (len(texts_for_embedding) + batch_size - 1) // batch_size
        
        for i in range(0, len(texts_for_embedding), batch_size):
            batch_num = i // batch_size + 1
            print(f"   Processing batch {batch_num}/{total_batches}")
            
            batch = texts_for_embedding[i:i + batch_size]
            batch_embeddings = self.embedder.encode(batch, convert_to_numpy=True)
            all_embeddings.append(batch_embeddings)
        
        # Combine all embeddings
        document_embeddings = np.vstack(all_embeddings)
        
        # Save embeddings
        print("💾 Saving document embeddings...")
        np.save(self.embeddings_path, document_embeddings)
        
        print(f"✅ Document embeddings created and saved")
        print(f"   - Shape: {document_embeddings.shape}")
        print(f"   - Saved to: {self.embeddings_path}")
        return True

    def verify_files(self):
        """Verify that all required files exist"""
        print("\n🔍 Verifying created files...")
        
        files_to_check = [
            (self.metadata_path, "Metadata JSON"),
            (self.bm25_index_path, "BM25 Index"),
            (self.bm25_docs_path, "BM25 Documents"),
            (self.embeddings_path, "Document Embeddings")
        ]
        
        all_exist = True
        for filepath, description in files_to_check:
            if os.path.exists(filepath):
                size = os.path.getsize(filepath) / (1024 * 1024)  # Size in MB
                print(f"   ✅ {description}: {filepath} ({size:.2f} MB)")
            else:
                print(f"   ❌ {description}: {filepath} (NOT FOUND)")
                all_exist = False
        
        return all_exist

    def create_all(self):
        """Create all required files for RAG pipeline"""
        print("="*60)
        print("🚀 CREATING METADATA, EMBEDDINGS AND INDICES")
        print("="*60)
        
        # Step 1: Create metadata from markdown files
        print("\n" + "="*60)
        print("STEP 1: CREATING METADATA FROM MARKDOWN FILES")
        print("="*60)
        
        if not self.create_metadata_from_markdown():
            print("\n❌ Failed to create metadata. Stopping.")
            return False
        
        # Step 2: Create BM25 index
        print("\n" + "="*60)
        print("STEP 2: CREATING BM25 INDEX")
        print("="*60)
        
        if not self.create_bm25_index():
            print("\n❌ Failed to create BM25 index. Stopping.")
            return False
        
        # Step 3: Create embeddings
        print("\n" + "="*60)
        print("STEP 3: CREATING EMBEDDINGS")
        print("="*60)
        
        if not self.create_embeddings():
            print("\n❌ Failed to create embeddings. Stopping.")
            return False
        
        # Step 4: Verify all files
        print("\n" + "="*60)
        print("STEP 4: VERIFICATION")
        print("="*60)
        
        success = self.verify_files()
        
        if success:
            print("\n✅ All files created successfully!")
            print("🎉 You can now run rag.py to start the RAG pipeline")
        else:
            print("\n⚠️  Some files are missing. Please check the errors above.")
        
        return success


def main():
    """Main function to create metadata, embeddings and indices"""
    
    # Configuration
    markdown_directory = "outputs"  # Base directory (outputs/{folder}/{folder}-referenced.md)
    metadata_path = "scibert_metadata.json"
    
    print("="*60)
    print("📚 METADATA AND EMBEDDING CREATOR FOR RAG PIPELINE")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"   Markdown Directory: {markdown_directory}")
    print(f"   Expected pattern: outputs/{{folder}}/{{folder}}-referenced.md")
    print(f"   Metadata File: {metadata_path}")
    print("\n" + "="*60)
    
    # Create the creator
    creator = MetadataAndEmbeddingCreator(
        markdown_directory=markdown_directory,
        metadata_path=metadata_path
    )
    
    # Check if files already exist
    files_exist = (
        os.path.exists(creator.metadata_path) and
        os.path.exists(creator.bm25_index_path) and
        os.path.exists(creator.bm25_docs_path) and
        os.path.exists(creator.embeddings_path)
    )
    
    if files_exist:
        print("\n⚠️  WARNING: Files already exist!")
        print("This will overwrite existing files:")
        print(f"   - {creator.metadata_path}")
        print(f"   - {creator.bm25_index_path}")
        print(f"   - {creator.bm25_docs_path}")
        print(f"   - {creator.embeddings_path}")
        
        response = input("\nDo you want to continue? (yes/no): ").strip().lower()
        if response not in ['yes', 'y']:
            print("❌ Operation cancelled.")
            return
    
    # Create all files
    success = creator.create_all()
    
    if success:
        print("\n" + "="*60)
        print("✅ PIPELINE READY!")
        print("="*60)
        print("\nNext steps:")
        print("1. Run your RAG pipeline: python rag.py")
        print("2. The system will use the generated files automatically")
        print("\n" + "="*60)


if __name__ == "__main__":
    main()