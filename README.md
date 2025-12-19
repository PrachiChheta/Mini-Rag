## Setup Instructions

1. Place the **`Mini-Rag`** folder in **Google Drive**.
2. Open **`Mini_Rag.ipynb`** in **Google Colab**.
3. Change the runtime to **T4**.
4. Run all cells.
   - When running:

     !pip install -r requirements.txt
     if it asks to **restart runtime**, do so.
   - After restarting, **run all cells from the beginning except the `!pip install -r requirements.txt` cell**.
5. Run the main routing script:
   !python rag.py

6. Click **Public URL** to open the **Gradio UI**.

---

## Folder Structure (`Mini-Rag`)

- **outputs/** → Markdown files generated from PDFs (processed chunks).
- **all_chunks.json** → Extracted text chunks with metadata, images, tables.
- **bm25_documents.pkl / bm25_index.pkl** → Pickle files for BM25 retrieval.
- **chunks.py** → Splits PDFs/text into sections or paragraphs.
- **embedding.py / document_embedding.py** → Creates embeddings for chunks for semantic search.
- **markdown.py** → Converts chunks to markdown files in `outputs/`.
- **model.py** → Loads LLaVA/LLaMA model for multimodal input.
- **rag.py** → Core RAG pipeline (retrieval + generation).
- **requirements.txt** → Python dependencies.
- **pdf_processing.log** → Logs PDF processing steps and errors.
- **scibert_metadata.json** → Metadata for SciBERT embeddings.

<img width="1898" height="899" alt="image" src="https://github.com/user-attachments/assets/24615524-5505-4928-9a89-9fe20caacb73" />
<img width="1899" height="592" alt="image" src="https://github.com/user-attachments/assets/cacc1482-ee9c-49aa-a0b8-405dbe506717" />

