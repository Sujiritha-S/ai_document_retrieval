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


# In[64]:


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


# In[65]:


transform = transforms.Compose([
    transforms.Resize((128, 128)),             # smaller images → faster CPU run
    transforms.Grayscale(num_output_channels=3),   # ensure 3 channels
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2), 
    transforms.ToTensor(),
])


# In[66]:


train_dir = r"E:\Mini Project\rvl_png\train"
val_dir   = r"E:\Mini Project\rvl_png\val"

train_ds = datasets.ImageFolder(train_dir, transform=transform)
val_ds   = datasets.ImageFolder(val_dir, transform=transform)

print("Full train samples:", len(train_ds))
print("Full val samples:", len(val_ds))
print("Classes:", train_ds.classes)


# In[67]:


# pick 32 training images, 8 validation images
mini_train_ds = Subset(train_ds, list(range(800)))
mini_val_ds   = Subset(val_ds, list(range(160)))

train_loader = DataLoader(mini_train_ds, batch_size=16, shuffle=True)
val_loader   = DataLoader(mini_val_ds, batch_size=8, shuffle=False)

print("Mini train samples:", len(mini_train_ds))
print("Mini val samples:", len(mini_val_ds))


# In[68]:


device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

num_classes = 16  # for RVL-CDIP small dataset

# Use lighter MobileNetV2 for CPU-friendly demo
model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

model = model.to(device)

for param in model.features.parameters():
    param.requires_grad = False


# In[69]:


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)


# In[70]:


#model.train()
#images, labels = next(iter(train_loader))
#images, labels = images.to(device), labels.to(device)

#optimizer.zero_grad()
#outputs = model(images)
#loss = criterion(outputs, labels)
#loss.backward()
#optimizer.step()

#print("Mini training done. Sample outputs shape:", outputs.shape)

num_epochs = 10  # can be 1 or 2
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


# In[71]:


model.eval()
with torch.no_grad():
    images, labels = next(iter(val_loader))
    images, labels = images.to(device), labels.to(device)
    outputs = model(images)
    _, preds = torch.max(outputs, 1)
    
print("Predicted classes:", [train_ds.classes[i] for i in preds])
print("Actual classes:   ", [train_ds.classes[i] for i in labels])


# In[77]:


import cv2
import numpy as np

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
                img_np = np.array(img)
                if len(img_np.shape) == 3:
                    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
                _, thresh = cv2.threshold(img_np, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                text = pytesseract.image_to_string(thresh)

                texts.append(text)
                labels.append(idx)
                count += 1
    return texts, labels, classes

train_texts, train_labels, classes = extract_text_from_folder(r"E:\Mini Project\rvl_png\train", num_per_class = 30)
val_texts, val_labels, _ = extract_text_from_folder(r"E:\Mini Project\rvl_png\val", num_per_class = 15)


# In[78]:


#pip install nltk
import nltk


# In[79]:


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


# In[80]:


#Convert Text To Embeddings

# Use a small, fast model for demo
model_embed = SentenceTransformer('msmarco-distilbert-base-dot-prod-v3')  

train_embeddings = model_embed.encode(train_texts)
val_embeddings   = model_embed.encode(val_texts)


# In[81]:


#Train A Mini-Classifier

#clf = LogisticRegression(max_iter=1000)
#clf.fit(train_embeddings, train_labels)

clf = SVC(kernel='linear')  # simple SVM
clf.fit(train_embeddings, train_labels)


# In[82]:


#Evaluate On Validation Set

val_preds = clf.predict(val_embeddings)
print(classification_report(val_labels, val_preds, target_names=classes))


# In[83]:


import re

def clean_ocr(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)   # remove symbols
    text = re.sub(r'\s+', ' ', text).strip()   # remove extra spaces
    return text


# In[84]:


import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Convert lists to arrays
train_embeddings_np = np.array(train_embeddings)
val_embeddings_np   = np.array(val_embeddings)

# Store texts together
all_texts = train_texts + val_texts
all_embeddings = np.vstack([train_embeddings_np, val_embeddings_np])

print("Embeddings DB shape:", all_embeddings.shape)


# In[85]:


def retrieve_top_k(query, k=3):
    # Embed the query
    q_emb = model_embed.encode([query])
    
    # Cosine similarity
    sims = cosine_similarity(q_emb, all_embeddings)[0]
    
    # Get top-k indices
    idx = [i for i in np.argsort(sims)[::-1] if sims[i] > 0.3][:k]

    # FILTER: Only keep docs above 0.25 similarity
    idx = [i for i in idx if sims[i] > 0.25][:k]
    
    results = []
    for i in idx:
        results.append({
            "similarity": float(sims[i]),
            "text": all_texts[i]
        })
    return results


# In[86]:


query = "invoice total amount and payment details"
top_docs = retrieve_top_k(query, k=3)

for i, doc in enumerate(top_docs):
    print(f"\n--- Result {i+1} (score={doc['similarity']:.4f}) ---")
    print(doc["text"][:500])   # show first 500 chars


# In[87]:


from transformers import pipeline

generator = pipeline("text-generation", model="distilgpt2")

def rag_answer(query):
    docs = retrieve_top_k(query, k=3)
    
    if not docs or len(docs[0]["text"].strip()) < 30:
        return "The system could not find relevant information in the documents."
    
    context = "\n\n".join([clean_ocr(doc['text']) for doc in docs])
    
    if "payment" not in context.lower() and "terms" not in context.lower():
        return "No payment terms are visible in the extracted invoice text."
    
    prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
    
    return generator(
        prompt,
        max_new_tokens=200, 
        truncation=True,         # FIX WARNING
        num_return_sequences=1,
        repetition_penalty=1.8,    # NEW
        no_repeat_ngram_size=3
    )[0]['generated_text']


# In[88]:


response = rag_answer("What does the invoice say about payment terms?")
print(response)


# In[ ]:




