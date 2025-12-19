#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""import cv2
import os

train_in = r"E:\Mini Project\rvl-cdip-small-200\train"
val_in   = r"E:\Mini Project\rvl-cdip-small-200\val"

train_out = r"E:\Mini Project\rvl_png\train"
val_out   = r"E:\Mini Project\rvl_png\val"

os.makedirs(train_out, exist_ok=True)
os.makedirs(val_out, exist_ok=True)

def convert_with_cv(input_path, output_path):
    for cls in os.listdir(input_path):
        cls_in = os.path.join(input_path, cls)
        cls_out = os.path.join(output_path, cls)
        os.makedirs(cls_out, exist_ok=True)

        for file in os.listdir(cls_in):
            if file.lower().endswith((".tif", ".tiff", ".png", ".jpg", ".jpeg")):
                in_path = os.path.join(cls_in, file)
                img = cv2.imread(in_path)     # SAFE loading (won’t kill kernel)
                if img is None:
                    continue

                out_name = os.path.splitext(file)[0] + ".png"
                out_path = os.path.join(cls_out, out_name)
                cv2.imwrite(out_path, img)

convert_with_cv(train_in, train_out)
convert_with_cv(val_in, val_out)

print("Conversion completed with OpenCV.")
"""


# In[1]:


import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
import os
import pytesseract
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import torch
from PIL import Image
from transformers import PreTrainedModel
from sklearn.svm import SVC


# In[2]:


transform = transforms.Compose([
    transforms.Resize((224, 224)),             
    transforms.Grayscale(num_output_channels=3),   
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.3, contrast=0.3), 
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])


# In[3]:


train_dir = r"E:\Mini Project\rvl_png\train"
val_dir   = r"E:\Mini Project\rvl_png\val"

train_ds = datasets.ImageFolder(train_dir, transform=transform)
val_ds   = datasets.ImageFolder(val_dir, transform=transform)

print("Full train samples:", len(train_ds))
print("Full val samples:", len(val_ds))
print("Classes:", train_ds.classes)


# In[4]:


train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=16, shuffle=False)

print("Train samples:", len(train_ds))
print("Val samples:", len(val_ds))


# In[5]:


device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

num_classes = 16  # for RVL-CDIP small dataset

# Use lighter MobileNetV2 for CPU-friendly demo
model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

model = model.to(device)

for name, param in model.features.named_parameters():
    if int(name.split('.')[0]) < 7:  # freeze first 7 blocks
        param.requires_grad = False


# In[6]:


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0005)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)


# In[7]:


num_epochs = 10  
model.train()

for epoch in range(num_epochs):
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}")

    scheduler.step()


# In[8]:


model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

from sklearn.metrics import accuracy_score, classification_report
import numpy as np

all_labels_np = np.array(all_labels)
all_preds_np  = np.array(all_preds)

print("Image classification accuracy:", accuracy_score(all_labels_np, all_preds_np))
print(classification_report(all_labels_np, all_preds_np, target_names=train_ds.classes))


# In[ ]:





# In[9]:


import cv2
import numpy as np

def ocr_preprocess(img_path):
    from PIL import Image

    img = Image.open(img_path)
    img_np = np.array(img)

    if len(img_np.shape) == 3:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    # Simple global threshold (original version)
    _, img_np = cv2.threshold(img_np, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Step 3: OCR
    text = pytesseract.image_to_string(img_np)
    return text


#Extract Text From Images

def extract_text_from_folder(folder, num_per_class=20):
    texts = []
    labels = []
    classes = sorted(os.listdir(folder))
    for idx, cls in enumerate(classes):
        cls_folder = os.path.join(folder, cls)
        count = 0
        for file in os.listdir(cls_folder):
            if count >= num_per_class:
                break
            if file.lower().endswith(".png"):
                img_path = os.path.join(cls_folder, file)
                img = Image.open(img_path)
                # --- OCR preprocessing ---
                text = ocr_preprocess(img_path)

                texts.append(text)
                labels.append(idx)
                count += 1
    return texts, labels, classes

train_texts, train_labels, classes = extract_text_from_folder(r"E:\Mini Project\rvl_png\train", num_per_class = 100)
val_texts, val_labels, _ = extract_text_from_folder(r"E:\Mini Project\rvl_png\val", num_per_class = 40)


# In[10]:


#pip install nltk
import nltk


# In[11]:


#Text Cleaning

import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)        # remove numbers
    text = re.sub(r'\W+', ' ', text)       # remove punctuation
    text = ' '.join([lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words])
    return text

train_texts = [clean_text(t) for t in train_texts]
val_texts   = [clean_text(t) for t in val_texts]


# In[12]:


#Convert Text To Embeddings

# Use a small, fast model for demo
model_embed = SentenceTransformer('paraphrase-mpnet-base-v2')  

train_embeddings = model_embed.encode(train_texts)
val_embeddings   = model_embed.encode(val_texts)


# In[13]:


#Train A Mini-Classifier
from sklearn.ensemble import RandomForestClassifier
#clf = LogisticRegression(max_iter=1000)
#clf.fit(train_embeddings, train_labels)

clf = SVC(kernel='linear')  # simple SVM
clf.fit(train_embeddings, train_labels)

# --- RandomForest ---
clf_rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
clf_rf.fit(train_embeddings, train_labels)


# In[14]:


#Evaluate On Validation Set
from sklearn.metrics import accuracy_score, classification_report

val_preds = clf.predict(val_embeddings)
print("Text classification: SVM accuracy:", accuracy_score(val_labels, val_preds))
print(classification_report(val_labels, val_preds, target_names=classes))

val_preds_rf  = clf_rf.predict(val_embeddings)
print("Text Classification: RandomForest Accuracy:", accuracy_score(val_labels, val_preds_rf))
print(classification_report(val_labels, val_preds_rf, target_names=classes))


# In[ ]:





# In[63]:


import re

def clean_ocr(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)   # remove symbols
    text = re.sub(r'\s+', ' ', text).strip()   # remove extra spaces
    return text


# In[64]:


import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Convert lists to arrays
train_embeddings_np = np.array(train_embeddings)
val_embeddings_np   = np.array(val_embeddings)

# Store texts together
all_texts = train_texts + val_texts
all_embeddings = np.vstack([train_embeddings_np, val_embeddings_np])

print("Embeddings DB shape:", all_embeddings.shape)


# In[65]:


def retrieve_top_k(query, k=3):
    # Embed the query
    q_emb = model_embed.encode([query])
    
    # Cosine similarity
    sims = cosine_similarity(q_emb, all_embeddings)[0]
    
    idx = np.argsort(sims)[::-1][:k]
    
    results = []
    for i in idx:
         # Determine class from index
        if i < len(train_texts):
            cls = classes[train_labels[i]]   # train label
        else:
            cls = classes[val_labels[i - len(train_texts)]]  # val label
        
        results.append({
            "similarity": float(sims[i]),
            "text": all_texts[i],
            "class": cls
        })
    return results


# In[66]:


query = "invoice total amount and payment details"
top_docs = retrieve_top_k(query, k=3)

for i, doc in enumerate(top_docs):
    print(f"\n--- Result {i+1} (score={doc['similarity']:.4f}) ---")
    print(doc["text"][:500])   # show first 500 chars


# In[67]:


#Method 2
def format_payment_answer(sentences, max_sentences=3):
    cleaned = []
    for s in sentences:
        s = s.strip()
        if len(s) > 25 and s not in cleaned:
            cleaned.append(s)
    return ". ".join(cleaned[:max_sentences]) + "."


# In[68]:


def rag_answer(query):
    docs = retrieve_top_k(query, k=3)

    if not docs:
        return "No relevant invoice documents were found."

    extracted = []
    for doc in docs:
        clean_text = clean_ocr(doc["text"])
        extracted.extend(extract_payment_sentences(clean_text))

    if not extracted:
        return "The invoice does not clearly specify payment terms."

    return format_payment_answer(extracted)


# In[56]:


#Method 1

from transformers import pipeline

generator = pipeline("text-generation", model="distilgpt2")

def rag_answer(query):
    docs = retrieve_top_k(query, k=3)
    
    if not docs or len(docs[0]["text"].strip()) < 30:
        return "The system could not find relevant information in the documents."
    
    context = "\n\n".join([
    clean_ocr(doc['text'])[:800]  # limit noise
    for doc in docs
    ])

    
    prompt = (
    "You are an assistant that answers questions using only the document text below.\n\n"
    f"Document:\n{context}\n\n"
    f"Question: {query}\n\n"
    "Answer in one or two clear sentences:"
    )

    
    return generator(
        prompt,
        max_new_tokens=120, 
        truncation=True,         # FIX WARNING
        num_return_sequences=1,
        repetition_penalty=1.5,    # NEW
        no_repeat_ngram_size=3
    )[0]['generated_text']


# In[69]:


response = rag_answer("What does the invoice say about payment terms?")
print(response)


# In[70]:


eval_queries = [
    {"query": "What are the payment terms in the invoice?", "expected_class": "invoice"},
    {"query": "How should the invoice be paid?", "expected_class": "invoice"},
    {"query": "What is the total amount due?", "expected_class": "invoice"},
    {"query": "Research publication payment details", "expected_class": "scientific_report"},
]


# In[71]:


def evaluate_retrieval_recall(eval_queries, k=3):
    correct = 0

    for q in eval_queries:
        results = retrieve_top_k(q["query"], k=k)
        retrieved_classes = [doc["class"] for doc in results]

        print("\nQuery:", q["query"])
        print("Retrieved classes:", retrieved_classes)

        if q["expected_class"] in retrieved_classes:
            correct += 1

    recall_at_k = correct / len(eval_queries)
    return recall_at_k


# In[72]:


recall_3 = evaluate_retrieval_recall(eval_queries, k=3)
print("\nRetrieval Recall@3:", recall_3)


# In[ ]:




