# AI-Powered Document Classification and Retrieval System

This project demonstrates an end-to-end pipeline for understanding scanned documents (invoices, receipts, etc.) by combining **image classification, OCR, and retrieval-augmented generation (RAG)**. It showcases how vision and NLP techniques can be integrated to extract, retrieve, and answer queries from document images.

## Key Features
- **Image preprocessing & classification**: Convert document images to PNG and classify them using a lightweight **MobileNetV2**.
- **Text extraction (OCR)**: Extract text from images using **Tesseract** with preprocessing (grayscale, thresholding).
- **Text embeddings & retrieval**: Embed extracted text using **Sentence Transformers (`paraphrase-mpnet-base-v2`)** and retrieve relevant information using cosine similarity.
- **RAG-style QA**: Generate answers to user queries by combining retrieved text with a **GPT-style generator** (optional) or rule-based extraction for invoices.
- **Evaluation metrics included**: Quantitative evaluation of each component to demonstrate reliability.

## Dataset & Scope
- Subset of the **RVL-CDIP** dataset used:
  - 16 document classes
  - **Image classification**: 800 training images (50 per class) and 160 validation images (10 per class)
  - **OCR & retrieval subset**: 30 training images per class and 15 validation images per class used for text embeddings

## Component Evaluation
### Component 1 — Image Classification (MobileNetV2)
- **Validation accuracy:** 70.15%
- CPU-trained, 16-class evaluation
- Training loss decreased steadily from **1.895 → 0.370**
- **Role:** Secondary signal

### Component 2 — Text Classification (Embeddings + Classifier)
- **Embedding:** `paraphrase-mpnet-base-v2`
- **Classifier:** Linear SVM
- **Accuracy:** ~52–53% on validation set
- **Role:** Secondary signal, not primary

### Component 3 — Text Retrieval (Primary Signal)
- **Retrieval:** Cosine similarity over `paraphrase-mpnet-base-v2` embeddings
- **Top-k:** k = 3
- **Evaluation:** Recall@3 ≈ 0.75 on evaluation queries
- **Role:** Primary signal for extracting invoice/payment information
- **Notes:** Works reliably for invoices; less accurate for rare/less-represented document types

## Workflow
1. Convert raw images to PNG format.  
2. Train a MobileNetV2 classifier on document images.  
3. Extract and clean text from images using OCR.  
4. Generate embeddings for the extracted text.  
5. Retrieve top relevant documents for a query using cosine similarity.  
6. Generate answers using GPT-style model (or rule-based extraction for invoices).
```
Document Image → [Image Preprocessing] → [MobileNetV2 Classifier]
                         ↓
                  [OCR Extraction]
                         ↓
             [Text Cleaning & Embeddings]
                         ↓
        [Text Retrieval using Cosine Similarity]
                         ↓
     [RAG-style Answer Generation / Rule-based QA]
```

## Example Use Case
**Query:** “What does the invoice say about payment terms?”  
**Output:** The system retrieves relevant text and generates a concise answer from the document content.

## Technologies & Libraries
- **Deep Learning:** PyTorch, torchvision, MobileNetV2  
- **OCR & Image Processing:** OpenCV, PIL, pytesseract  
- **NLP & Embeddings:** Sentence Transformers, scikit-learn, transformers (GPT-style generation)  
- **Python Utilities:** numpy, nltk, re  
