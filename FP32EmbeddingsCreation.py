import numpy as np
import torch
import csv
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Paths
script_dir = Path(__file__).resolve().parent
dataset_dir = script_dir / "DataSet"
dataset_dir.mkdir(parents=True, exist_ok=True)

# Heuristics for column detection
TEXT_COL_CANDIDATES = ['text', 'message', 'v2', 'content', 'body', 'review']
LABEL_COL_CANDIDATES = ['label', 'category', 'target', 'sentiment', 'v1', 'class']

def detect_columns(fieldnames):
    fields_lower = [f.lower() for f in (fieldnames or [])]
    text_col = None
    label_col = None
    for c in TEXT_COL_CANDIDATES:
        if c in fields_lower:
            text_col = fieldnames[fields_lower.index(c)]
            break
    for c in LABEL_COL_CANDIDATES:
        if c in fields_lower:
            label_col = fieldnames[fields_lower.index(c)]
            break
    return text_col, label_col

def map_labels_to_ints(labels_list):
    # Try integer cast first; if any fail, map strings to ints
    mapped = []
    label_to_int = {}
    next_int = 0
    for lbl in labels_list:
        try:
            mapped.append(int(lbl))
            continue
        except (ValueError, TypeError):
            pass
        key = str(lbl).strip().lower()
        if key not in label_to_int:
            label_to_int[key] = next_int
            next_int += 1
        mapped.append(label_to_int[key])
    return np.array(mapped, dtype=np.int64)

@torch.no_grad()
def bert_embed(texts, tokenizer, bert_model, max_len=512, batch_size=32):
    embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        toks = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_len).to(device)
        if "token_type_ids" in toks:
            del toks["token_type_ids"]
        out = bert_model(**toks).last_hidden_state  # [B, L, 768]
        cls = out[:, 0, :]                          # [B, 768]
        embs.append(cls.cpu().numpy())
    return np.vstack(embs).astype(np.float32)

# Load BERT once
bert_model_name = "bert-base-uncased"
print(f"Loading {bert_model_name}...")
tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
bert_model = AutoModel.from_pretrained(bert_model_name).to(device)
for p in bert_model.parameters():
    p.requires_grad = False
bert_model.eval()

# Process all CSV files in DataSet
csv_files = sorted(dataset_dir.glob("*.csv"))
if not csv_files:
    print(f"No CSV files found in {dataset_dir}")
else:
    print(f"Found {len(csv_files)} CSV file(s) in {dataset_dir}")

for csv_path in csv_files:
    print(f"\nProcessing: {csv_path.name}")
    texts = []
    labels = []
    encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 'utf-8-sig']
    file_loaded = False

    for encoding in encodings:
        try:
            with open(csv_path, 'r', encoding=encoding, errors='replace', newline='') as f:
                reader = csv.DictReader(f)
                text_col, label_col = detect_columns(reader.fieldnames)
                if not text_col or not label_col:
                    raise ValueError("Could not detect text/label columns")
                for row in reader:
                    text = (row.get(text_col) or "").strip()
                    label = (row.get(label_col) or "").strip()
                    if text:
                        texts.append(text)
                        labels.append(label)
            file_loaded = True
            print(f"  Loaded with encoding: {encoding}")
            break
        except Exception as e:
            texts = []
            labels = []
            continue

    if not file_loaded:
        print(f"  Skipped (could not load with known encodings): {csv_path.name}")
        continue

    if len(texts) == 0 or len(labels) == 0:
        print("  Skipped (no records after parsing).")
        continue

    labels = map_labels_to_ints(labels)
    print(f"  Samples: {len(texts)} | Label classes: {int(labels.max())+1 if labels.size else 0}")

    # Split dataset into train/val/test: 80/20 then 10% of train to val
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels if len(set(labels)) > 1 else None
    )
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_texts, train_labels, test_size=0.1, random_state=42,
        stratify=train_labels if len(set(train_labels)) > 1 else None
    )

    print(f"  Split -> Train: {len(train_texts)}, Val: {len(val_texts)}, Test: {len(test_texts)}")

    # Create embeddings
    print("  Creating embeddings...")
    print("    Training embeddings...")
    train_emb = bert_embed(train_texts, tokenizer, bert_model)
    print("    Validation embeddings...")
    val_emb = bert_embed(val_texts, tokenizer, bert_model)
    print("    Test embeddings...")
    test_emb = bert_embed(test_texts, tokenizer, bert_model)

    # Dataset name: exactly the CSV stem from DataSet folder
    ds_name = csv_path.stem

    # Output directory: DataSet/embeddings_<datasetname>
    out_dir = dataset_dir / f"embeddings"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save embeddings (standardized names)
    print(f"  Saving embeddings to {out_dir} ...")
    np.save(out_dir / 'train_emb.npy', train_emb)
    np.save(out_dir / 'val_emb.npy', val_emb)
    np.save(out_dir / 'test_emb.npy', test_emb)

    # Save labels (standardized names)
    print("  Saving labels...")
    np.save(out_dir / 'train_labels.npy', np.array(train_labels, dtype=np.int64))
    np.save(out_dir / 'val_labels.npy', np.array(val_labels, dtype=np.int64))
    np.save(out_dir / 'test_labels.npy', np.array(test_labels, dtype=np.int64))

    print("  Done.")
    print(f"    Train embeddings: {train_emb.shape} | labels: {len(train_labels)}")
    print(f"    Val embeddings  : {val_emb.shape} | labels: {len(val_labels)}")
    print(f"    Test embeddings : {test_emb.shape} | labels: {len(test_labels)}")

