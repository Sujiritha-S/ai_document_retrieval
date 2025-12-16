# AI-Powered Document Classification and Retrieval System

This project demonstrates an end-to-end pipeline for understanding scanned documents (invoices, receipts, etc.) by combining **image classification, OCR, and retrieval-augmented generation (RAG)**.  
It showcases how vision and NLP techniques can be integrated to extract, retrieve, and answer queries from document images.

## Key Features

- **Image preprocessing & classification:** Convert document images to PNG and classify them using a lightweight MobileNetV2.  
- **Text extraction (OCR):** Extract text from images using Tesseract with preprocessing (grayscale, thresholding).  
- **Text embeddings & retrieval:** Embed extracted text using Sentence Transformers and retrieve relevant information using cosine similarity.  
- **RAG-style QA:** Generate answers to user queries by combining retrieved text with a GPT-style generator.

## Dataset & Scope

- Subset of the RVL-CDIP dataset used for this project:  
  - 16 document classes  
  - Image classification: 800 training images (50 per class) and 160 validation images (10 per class)  
  - OCR & retrieval subset: 30 training images per class and 15 validation images per class used for text embeddings  
- Image classification accuracy on this mini dataset: ~50%  
- CPU-friendly pipeline designed for efficient experimentation

## Workflow

1. Convert raw images to PNG format.  
2. Train a MobileNetV2 classifier on document images.  
3. Extract and clean text from images using OCR.  
4. Generate embeddings for the extracted text.  
5. Retrieve top relevant documents for a query and generate answers using a GPT-style model.

## Project Pipeline

Document Image → [Image Preprocessing] → [MobileNetV2 Classifier]  
                         ↓  
                   [OCR Extraction]  
                         ↓  
                 [Text Cleaning & Embeddings]  
                         ↓  
          [Text Retrieval using Cosine Similarity]  
                         ↓  
          [RAG-style Answer Generation (GPT model)]

## Example Use Case

**Query:** “What does the invoice say about payment terms?”  
**Output:** The system retrieves relevant text and generates a concise answer from the document content.  

## Technologies & Libraries

- **Deep Learning:** PyTorch, torchvision, MobileNetV2  
- **OCR & Image Processing:** OpenCV, PIL, pytesseract  
- **NLP & Embeddings:** Sentence Transformers, scikit-learn, transformers (GPT-style generation)  
- **Python Utilities:** numpy, nltk, re

---

This project demonstrates a practical approach to combining computer vision and NLP techniques for document understanding and question answering.

