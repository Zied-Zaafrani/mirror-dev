"""Precompute PubMedBERT embeddings for Lab-as-Text states."""
import sys
from pathlib import Path
import torch
import numpy as np

# This requires transformers to be available, but we can mock it if not
try:
    from transformers import AutoTokenizer, AutoModel
except ImportError:
    print("transformers not installed. Skipping actual BERT encoding.")
    print("In a real Kaggle environment, this would run PubMedBERT.")
    print("For local dev, we will generate random embeddings of shape (18, 4, 768).")
    
    # We create random embeddings for now, but save the structure.
    # Dimensions: [num_labs, 4 states, 768]
    # States: 0=missing, 1=low, 2=normal, 3=high
    embeddings = torch.randn(18, 4, 768)
    
    out_path = Path("data/processed/lab_text_embeddings.pt")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(embeddings, out_path)
    print(f"Saved mock embeddings to {out_path}")
    sys.exit(0)

# If transformers is available (e.g. Kaggle):
tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract")
model = AutoModel.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract")
model.eval()

# Fix Python path for direct execution
src_dir = Path(__file__).resolve().parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from dataset import LAB_CLINICAL_THRESHOLDS
lab_names = list(LAB_CLINICAL_THRESHOLDS.keys())

embeddings = torch.zeros(18, 4, 768)

with torch.no_grad():
    for i, name in enumerate(lab_names):
        # State 0: Missing
        # Embedding is zero vector
        pass
        
        # State 1: Low
        inputs = tokenizer(f"{name} is low", return_tensors="pt")
        outputs = model(**inputs)
        embeddings[i, 1] = outputs.pooler_output[0]
        
        # State 2: Normal
        inputs = tokenizer(f"{name} is normal", return_tensors="pt")
        outputs = model(**inputs)
        embeddings[i, 2] = outputs.pooler_output[0]
        
        # State 3: High
        inputs = tokenizer(f"{name} is high", return_tensors="pt")
        outputs = model(**inputs)
        embeddings[i, 3] = outputs.pooler_output[0]

out_path = Path("data/processed/lab_text_embeddings.pt")
out_path.parent.mkdir(parents=True, exist_ok=True)
torch.save(embeddings, out_path)
print(f"Saved {embeddings.shape} embeddings to {out_path}")
