import json
import numpy as np
import pickle
import os
from PIL import Image
import torch
import gradio as gr
from sentence_transformers import SentenceTransformer, CrossEncoder
from model import load_llava_model_4bit
import re
from typing import List, Dict, Any, Tuple, Optional
from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import wordpunct_tokenize
from nltk.corpus import stopwords

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')


class SimpleRAGPipeline:
    def __init__(self):
        print("🚀 Initializing RAG Pipeline...")
        
        # Load LLaVA model
        print("📥 Loading LLaVA model...")
        self.processor, self.model = load_llava_model_4bit()
        
        # Load SciBERT embedder (cased version - same as create_embeddings.py)
        print("📥 Loading SciBERT embedding model (cased)...")
        try:
            self.embedder = SentenceTransformer("allenai/scibert_scivocab_cased")
            print("✅ SciBERT (cased) model loaded successfully!")
        except Exception as e:
            print(f"❌ Failed to load SciBERT model: {str(e)}")
            print("🔄 Falling back to MiniLM model...")
            self.embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        
        # Load Cross-Encoder for reranking
        print("📥 Loading Cross-Encoder for reranking...")
        try:
            self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
            print("✅ Cross-Encoder loaded successfully!")
        except Exception as e:
            print(f"❌ Failed to load Cross-Encoder: {str(e)}")
            self.cross_encoder = None
        
        # Load metadata
        print("📚 Loading metadata...")
        with open("scibert_metadata.json", "r") as f:
            self.metadata = json.load(f)
        print(f"✅ Loaded {len(self.metadata)} chunks from metadata")
        
        # Load document embeddings
        print("📊 Loading document embeddings...")
        self.document_embeddings = np.load("document_embeddings.npy")
        print(f"✅ Loaded embeddings with shape: {self.document_embeddings.shape}")
        
        # Load BM25 index
        print("📖 Loading BM25 index...")
        with open("bm25_index.pkl", 'rb') as f:
            self.bm25 = pickle.load(f)
        with open("bm25_documents.pkl", 'rb') as f:
            self.bm25_documents = pickle.load(f)
        print(f"✅ BM25 index loaded with {len(self.bm25_documents)} documents")
        
        # Verify alignment
        if len(self.metadata) != len(self.document_embeddings) != len(self.bm25_documents):
            print(f"⚠️ WARNING: Size mismatch!")
            print(f"   Metadata: {len(self.metadata)}")
            print(f"   Embeddings: {len(self.document_embeddings)}")
            print(f"   BM25 docs: {len(self.bm25_documents)}")
        else:
            print(f"✅ All components aligned: {len(self.metadata)} documents")
        
        print("✅ RAG Pipeline initialized successfully!\n")

    def prepare_text_for_search(self, chunk: Dict) -> str:
        """
        Prepare text content from chunk for search indexing
        (Same method as in create_embeddings.py)
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

    def semantic_search(self, query: str, top_k: int = 20) -> List[Tuple[int, float]]:
        """
        Semantic search using SciBERT embeddings
        Returns list of (index, score) tuples
        """
        print(f"🔍 Semantic search (SciBERT)...")
        
        # Encode query using same embedder
        query_embedding = self.embedder.encode([query], convert_to_numpy=True)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)  # Normalize
        
        # Normalize document embeddings
        doc_embeddings_norm = self.document_embeddings / np.linalg.norm(
            self.document_embeddings, axis=1, keepdims=True
        )
        
        # Calculate cosine similarity
        similarities = np.dot(doc_embeddings_norm, query_embedding.T).flatten()
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = [(int(idx), float(similarities[idx])) for idx in top_indices]
        
        print(f"   Found {len(results)} results")
        return results

    def bm25_search(self, query: str, top_k: int = 20) -> List[Tuple[int, float]]:
        """
        BM25 keyword search
        Returns list of (index, score) tuples
        """
        print(f"🔍 BM25 keyword search...")
        
        # Tokenize query (same preprocessing as in create_embeddings.py)
        stop_words = set(stopwords.words('english'))
        query_tokens = wordpunct_tokenize(query.lower())
        query_tokens = [token for token in query_tokens if token.isalnum() and token not in stop_words]
        
        # Get BM25 scores
        bm25_scores = self.bm25.get_scores(query_tokens)
        
        # Get top-k indices
        top_indices = np.argsort(bm25_scores)[::-1][:top_k]
        
        results = [(int(idx), float(bm25_scores[idx])) for idx in top_indices if bm25_scores[idx] > 0]
        
        print(f"   Found {len(results)} results with non-zero scores")
        return results

    def reciprocal_rank_fusion(self, 
                               semantic_results: List[Tuple[int, float]], 
                               bm25_results: List[Tuple[int, float]], 
                               k: int = 60) -> List[Tuple[int, float]]:
        """
        Merge results using Reciprocal Rank Fusion (RRF)
        
        Args:
            semantic_results: List of (index, score) from semantic search
            bm25_results: List of (index, score) from BM25 search
            k: RRF constant (default 60)
            
        Returns:
            List of (index, rrf_score) sorted by RRF score
        """
        print(f"🔀 Applying Reciprocal Rank Fusion (k={k})...")
        
        rrf_scores = {}
        
        # Add semantic search results
        for rank, (idx, score) in enumerate(semantic_results, start=1):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (k + rank)
        
        # Add BM25 results
        for rank, (idx, score) in enumerate(bm25_results, start=1):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (k + rank)
        
        # Sort by RRF score
        merged_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        
        print(f"   Merged to {len(merged_results)} unique results")
        return merged_results

    def rerank_with_cross_encoder(self, 
                                  query: str, 
                                  candidates: List[Tuple[int, float]], 
                                  top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Rerank candidates using Cross-Encoder
        
        Args:
            query: User query
            candidates: List of (index, score) tuples
            top_k: Number of top results to return after reranking
            
        Returns:
            List of (index, rerank_score) sorted by rerank score
        """
        if not self.cross_encoder:
            print("⚠️ Cross-Encoder not available, returning original ranking")
            return candidates[:top_k]
        
        print(f"🔄 Reranking top {len(candidates)} candidates with Cross-Encoder...")
        
        # Prepare query-document pairs
        pairs = []
        indices = []
        
        for idx, score in candidates:
            if idx < len(self.metadata):
                chunk = self.metadata[idx]
                text = self.prepare_text_for_search(chunk)
                # Truncate text if too long (Cross-Encoder has token limits)
                text = text[:512]  # Keep first 512 characters
                pairs.append([query, text])
                indices.append(idx)
        
        # Get rerank scores
        rerank_scores = self.cross_encoder.predict(pairs)
        
        # Combine indices with rerank scores
        reranked_results = list(zip(indices, rerank_scores))
        reranked_results.sort(key=lambda x: x[1], reverse=True)
        
        print(f"   Reranked and returning top {min(top_k, len(reranked_results))} results")
        return reranked_results[:top_k]

    def hybrid_search_with_rerank(self, query: str, top_k: int = 10) -> Dict:
        """
        Main search function:
        1. Semantic search (SciBERT)
        2. BM25 keyword search
        3. Merge with RRF
        4. Rerank with Cross-Encoder
        
        Args:
            query: User query
            top_k: Number of final results to return
            
        Returns:
            Best chunk after reranking
        """
        print(f"\n{'='*60}")
        print(f"🔍 HYBRID SEARCH WITH RERANKING")
        print(f"Query: '{query}'")
        print(f"{'='*60}\n")
        
        # Step 1: Semantic search
        semantic_results = self.semantic_search(query, top_k=20)
        
        # Step 2: BM25 search
        bm25_results = self.bm25_search(query, top_k=20)
        
        # Step 3: Merge with RRF
        merged_results = self.reciprocal_rank_fusion(semantic_results, bm25_results, k=60)
        
        # Take top candidates for reranking (more candidates = better reranking)
        candidates_for_rerank = merged_results[:30]
        
        # Step 4: Rerank with Cross-Encoder
        final_results = self.rerank_with_cross_encoder(query, candidates_for_rerank, top_k=top_k)
        
        # Debug output
        print(f"\n📊 Top 5 Final Results:")
        for i, (idx, score) in enumerate(final_results[:5], start=1):
            chunk = self.metadata[idx]
            print(f"  {i}. Score: {score:.4f}")
            print(f"     Document: {chunk.get('pdf', 'Unknown')}")
            print(f"     Section: {chunk.get('section_title', 'No title')}")
            print(f"     Text preview: {chunk.get('text', '')[:100]}...")
            print()
        
        # Return best chunk
        if final_results:
            best_idx, best_score = final_results[0]
            print(f"🏆 SELECTED BEST CHUNK:")
            print(f"   Index: {best_idx}")
            print(f"   Rerank Score: {best_score:.4f}")
            return self.metadata[best_idx]
        
        return None

    def resize_image(self, image: Image.Image):
        """Resize image for LLaVA model"""
        patch_size = self.model.config.vision_config.patch_size
        shortest_edge = self.processor.image_processor.size.get("shortest_edge", 336)

        orig_w, orig_h = image.size
        scale = shortest_edge / min(orig_w, orig_h)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)

        new_w = (new_w // patch_size) * patch_size
        new_h = (new_h // patch_size) * patch_size

        return image.resize((new_w, new_h))

    def create_context(self, chunk: Dict) -> str:
        """Create context string from chunk"""
        if not chunk:
            return "No relevant content found."

        section_title = chunk.get("section_title", "Section")
        text = chunk.get("text", "").strip()
        pdf_name = chunk.get("pdf", "Unknown Document")

        context = f"Document: {pdf_name}\nSection: {section_title}\n\nContent:\n{text}\n"

        tables = chunk.get("tables", [])
        if tables:
            for table in tables:
                caption = table.get("caption", "")
                headers = table.get("headers", [])
                rows = table.get("rows", [])

                table_str = f"\n{caption}\n"
                if headers:
                    table_str += " | ".join(headers) + "\n"
                for row in rows:
                    table_str += " | ".join(row) + "\n"
                context += f"\nTable:\n{table_str}\n"

        return context.strip()

    def clean_answer(self, answer: str) -> str:
        """Clean up generated answer"""
        answer = re.sub(r'\[\d+\]', '', answer)
        answer = re.sub(r'\s+', ' ', answer).strip()
        answer = re.sub(r'^[,\s]*', '', answer)
        answer = re.sub(r'[,\s]*$', '', answer)
        answer = re.sub(r',\s*,', ',', answer)
        return answer

    def generate_answer(self, query: str, chunk: Dict) -> str:
        """Generate answer using LLaVA"""
        if not chunk:
            return "No relevant document section found for your query."

        context = self.create_context(chunk)
        image_paths = chunk.get("images", [])
        if isinstance(image_paths, str):
            image_paths = [image_paths]

        valid_images = []
        for img_path in image_paths:
            try:
                img = self.resize_image(Image.open(img_path).convert("RGB"))
                valid_images.append(img)
                print(f"✅ Loaded image: {img_path}")
            except Exception as e:
                print(f"❌ Failed to load image {img_path}: {e}")

        if valid_images:
            print(f"🖼️ Generating answer with {len(valid_images)} images + text")
            answer = self.generate_with_images(query, context, valid_images)
            if "not provide enough information" in answer or len(answer) < 50:
                print("⚠️ Vision answer insufficient. Retrying with text only.")
                answer = self.generate_text_only(query, context)
            return self.clean_answer(answer)

        print("📝 No valid images - using text only")
        answer = self.generate_text_only(query, context)
        return self.clean_answer(answer)

    def generate_text_only(self, query: str, context: str) -> str:
        """Generate answer from text context only"""
        prompt = f"""Answer the question using ONLY the information provided in the context below.

CONTEXT:
{context}

QUESTION: {query}

CRITICAL INSTRUCTIONS:
• Answer ONLY using information present in the above context
• If the information is NOT in the context, clearly state "This information is not available in the provided context"
• DO NOT use any external knowledge or information beyond what is provided
• DO NOT make assumptions or add information not explicitly stated in the context
• Use ONLY the information provided in the reference document
• Structure your response with **bold text**, ## headers, and - bullet points
• Include specific details, data, and measurements when available from the context
• If tables are present in context, reference specific values
• DO NOT include any citation numbers, brackets, or reference markers
• DO NOT use phrases like "according to the document" or "the paper states"
• Write as if the information is established fact (since you're using authoritative sources)
• Use markdown formatting for clear structure
• Be concise but thorough
• Avoid giving introduction and conclusion, get to the main answer
• REMEMBER: If any part of the answer requires information NOT in the context, say so explicitly
Answer:

ANSWER:"""

        print(f"📝 Context length: {len(context)} characters")

        inputs = self.processor(text=prompt, return_tensors="pt")
        inputs = {k: v.to("cuda:0") if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

        if "attention_mask" not in inputs:
            inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])

        with torch.no_grad():
            output = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=512,
                do_sample=True,
                top_p=0.9,
                temperature=0.1,
                pad_token_id=self.processor.tokenizer.eos_token_id,
                eos_token_id=self.processor.tokenizer.eos_token_id,
                repetition_penalty=1.1
            )

        input_length = inputs["input_ids"].shape[1]
        response_tokens = output[0][input_length:]
        response = self.processor.tokenizer.decode(response_tokens, skip_special_tokens=True)

        answer = response.strip()
        answer = self.clean_answer(answer)

        return answer if answer and len(answer.strip()) > 0 else "Unable to generate a proper answer from the provided context."

    def generate_with_images(self, query: str, context: str, images: List[Image.Image]) -> str:
        """Generate answer with images and context"""
        image_tokens = "<image>" * len(images)
        prompt = f"""Answer the question using the provided images and text context.

{image_tokens}

CONTEXT:
{context}

QUESTION: {query}
CRITICAL INSTRUCTIONS:
• Answer ONLY using information present in the above context
• If the information is NOT in the context, clearly state "This information is not available in the provided context"
• DO NOT use any external knowledge or information beyond what is provided
• DO NOT make assumptions or add information not explicitly stated in the context
• Use ONLY the information provided in the reference document
• Structure your response with **bold text**, ## headers, and - bullet points
• Include specific details, data, and measurements when available from the context
• If tables are present in context, reference specific values
• DO NOT include any citation numbers, brackets, or reference markers
• DO NOT use phrases like "according to the document" or "the paper states"
• Write as if the information is established fact (since you're using authoritative sources)
• Use markdown formatting for clear structure
• Be concise but thorough
• Avoid giving introduction and conclusion, get to the main answer
• REMEMBER: If any part of the answer requires information NOT in the context, say so explicitly

SCIENTIFIC IMAGE ANALYSIS INSTRUCTIONS:

**MICROSCOPY IMAGES (TEM/SEM/AFM):**
- Identify the microscopy technique from image characteristics
- Note scale bar and magnification level
- Describe crystal structures, grain boundaries, defects, or morphology
- Identify specific features like dislocations, twins, precipitates, or interfaces
- Analyze crystalline vs amorphous regions
- Note any diffraction patterns or SAED
- Describe particle size, shape, and distribution

**TECHNICAL DIAGRAMS/SCHEMATICS:**
- Describe the system or process being illustrated
- Identify components, connections, and flow directions
- Note symbols, legends, or technical notations
- Explain the working principle or mechanism shown

**GRAPHS/CHARTS:**
- Identify axes (X and Y labels, units, scales)
- Describe plotted data, trends, clusters, or anomalies
- Interpret curves, colors, symbols, or time steps
- Compare multiple subplots if shown
- Mention variables or materials used

**MATERIAL CHARACTERIZATION:**
- Describe surface morphology, roughness, or texture
- Identify crystal phases, orientations, or structures
- Note compositional variations or elemental mapping
- Describe defects, cracks, or failure mechanisms

Do NOT include citation markers like [15], [26], etc.
If the image doesn't contain information relevant to the question, state this clearly.
Your answer must be **explainable** using information from the context.

ANSWER:"""

        inputs = self.processor(images=images, text=prompt, return_tensors="pt")
        inputs = {k: v.to("cuda:0") if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

        if "attention_mask" not in inputs:
            inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])

        with torch.no_grad():
            output = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=512,
                do_sample=True,
                top_p=0.9,
                temperature=0.1,
                pad_token_id=self.processor.tokenizer.eos_token_id,
                eos_token_id=self.processor.tokenizer.eos_token_id,
                repetition_penalty=1.1
            )

        input_length = inputs["input_ids"].shape[1]
        response_tokens = output[0][input_length:]
        response = self.processor.tokenizer.decode(response_tokens, skip_special_tokens=True)

        answer = response.strip()
        answer = self.clean_answer(answer)

        return answer if answer and len(answer.strip()) > 0 else "Unable to generate a proper answer from the provided content."

    def pipeline(self, user_query: str) -> Tuple[str, str, str, List[Optional[Image.Image]]]:
        """
        Main pipeline: search and generate answer
        Returns: (answer, pdf_name, text_content, loaded_images)
        """
        if not user_query.strip():
            return "Please enter a question.", "No PDF selected", "No text content", []

        try:
            print(f"\n{'='*60}")
            print(f"🔍 Processing query: '{user_query}'")
            print(f"{'='*60}\n")
            
            # Use hybrid search with reranking
            best_chunk = self.hybrid_search_with_rerank(user_query, top_k=10)

            if not best_chunk:
                return "No relevant information found in the documents.", "No PDF found", "No text content", []

            # Extract chunk information
            pdf_name = best_chunk.get("pdf", "Unknown Document")
            text_content = best_chunk.get("text", "No text content available")
            image_paths = best_chunk.get("images", [])
            
            # Ensure image_paths is a list
            if isinstance(image_paths, str):
                image_paths = [image_paths]
            
            # Load all valid images
            loaded_images = []
            for img_path in image_paths:
                try:
                    img = Image.open(img_path)
                    loaded_images.append(img)
                    print(f"✅ Loaded image: {img_path}")
                except Exception as e:
                    print(f"❌ Failed to load image {img_path}: {e}")
            
            # Generate answer
            answer = self.generate_answer(user_query, best_chunk)
            
            return answer, pdf_name, text_content, loaded_images

        except Exception as e:
            print(f"❌ Pipeline error: {e}")
            import traceback
            traceback.print_exc()
            return f"Error processing your query: {str(e)}", "Error", "Error", []


# ========== Gradio UI ==========
if __name__ == "__main__":
    rag_pipeline = SimpleRAGPipeline()

    def gradio_interface(user_query):
        answer, pdf_name, text_content, loaded_images = rag_pipeline.pipeline(user_query)
        return answer, pdf_name, text_content, loaded_images

    with gr.Blocks(title="Mini RAG Assistant") as interface:
        
        gr.Markdown("# Mini RAG Assistant")

        
        # Question Input
        with gr.Row():
            user_input = gr.Textbox(
                lines=3, 
                placeholder="Type your question here...", 
                label="Your Question"
            )
        
        with gr.Row():
            submit_btn = gr.Button("Ask Question", variant="primary")
        
        # Answer Section
        gr.Markdown("## Answer")
        with gr.Row():
            answer_output = gr.Markdown(
                value="",
                container=True,
                height=90
            )
        
        # Source Information Section
        gr.Markdown("## Source Information")
        
        with gr.Row():
            pdf_output = gr.Textbox(label="PDF Document", lines=1)
            
        with gr.Row():
            text_output = gr.Textbox(label="Text Content", lines=6)
        
        # Images Section
        gr.Markdown("## Related Images")
        with gr.Row():
            images_gallery = gr.Gallery(
                label="Images from Selected Chunk",
                show_label=True,
                columns=3,
                rows=2,
                object_fit="contain",
                height="auto"
            )
        
        # Connect the function
        submit_btn.click(
            fn=gradio_interface,
            inputs=[user_input],
            outputs=[answer_output, pdf_output, text_output, images_gallery]
        )
        
        # Also allow Enter key to submit
        user_input.submit(
            fn=gradio_interface,
            inputs=[user_input],
            outputs=[answer_output, pdf_output, text_output, images_gallery]
        )
    
    interface.launch(share=True)