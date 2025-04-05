import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
import os
import time
import json

# Disable wandb
os.environ["WANDB_DISABLED"] = "true"

# Track start time
start_time = time.time()

# Enable GPU usage check
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

# Load dataset from local path
df = pd.read_csv('WELFake_Dataset.csv')

# Display dataset information
print("Dataset shape:", df.shape)
print("\nClass distribution:")
print(df['label'].value_counts())
print("\nSample data:")
print(df.head())

# Clean data
df = df[['text', 'label']].dropna()
print(f"\nClean dataset shape: {df.shape}")

# Split data with stratification to maintain class balance
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['text'].tolist(), 
    df['label'].tolist(), 
    test_size=0.2,
    random_state=42,
    stratify=df['label']
)

print(f"Training examples: {len(train_texts)}")
print(f"Validation examples: {len(val_texts)}")

# Calculate class weights for handling imbalance
class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
class_weights_dict = {i: class_weights[i] for i in range(len(class_weights))}
print(f"Class weights: {class_weights_dict}")

# Convert to torch tensors for GPU use
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).cuda()

# Tokenization with optimized max_length
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
max_length = 128  # Optimized for speed while maintaining accuracy
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=max_length)
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=max_length)

# Dataset class
class FakeNewsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

train_dataset = FakeNewsDataset(train_encodings, train_labels)
val_dataset = FakeNewsDataset(val_encodings, val_labels)

# Load model
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)

# Optimized training arguments for faster training while maintaining accuracy
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=2,  # Reduced epochs but increased batch size
    per_device_train_batch_size=128,  # Increased batch size for faster training
    per_device_eval_batch_size=256,
    warmup_steps=50,  # Reduced warmup
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=50,
    evaluation_strategy="steps",  # Evaluate more frequently
    eval_steps=500,  # Evaluate every 500 steps
    save_strategy="steps",
    save_steps=500,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    fp16=True,  # Enable mixed precision
    gradient_accumulation_steps=1,
    report_to=None,
)

# Metrics function
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = accuracy_score(labels, predictions)
    return {'accuracy': accuracy}

# Create trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

# Train model
print("\nStarting training...")
trainer.train()

# Calculate training time
training_time = (time.time() - start_time) / 60
print(f"\nTraining completed in {training_time:.2f} minutes")

# Evaluate model
eval_results = trainer.evaluate()
print(f"Validation Accuracy: {eval_results['eval_accuracy']:.4f}")

# Create model export directory
export_dir = './fake_news_model'
os.makedirs(export_dir, exist_ok=True)

# Save model in multiple formats
print("\nSaving model in multiple formats...")

# 1. Save Hugging Face format
hf_model_path = os.path.join(export_dir, 'huggingface_model')
model.save_pretrained(hf_model_path)
tokenizer.save_pretrained(hf_model_path)

# 2. Save PyTorch format
torch.save(model.state_dict(), os.path.join(export_dir, 'model.pt'))

# 3. Save model metadata
model_config = {
    "model_type": "DistilBERT",
    "task": "fake_news_classification",
    "class_mapping": {0: "REAL", 1: "FAKE"},
    "max_length": max_length,
    "accuracy": eval_results['eval_accuracy'],
    "training_time_minutes": training_time
}

with open(os.path.join(export_dir, "model_metadata.json"), "w") as f:
    json.dump(model_config, f, indent=2)

# 4. Create inference script
inference_script = """
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

def load_model(model_path):
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
    model = DistilBertForSequenceClassification.from_pretrained(model_path)
    return tokenizer, model

def predict(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    predicted_class = torch.argmax(outputs.logits, dim=1).item()
    return "REAL" if predicted_class == 0 else "FAKE"

if __name__ == "__main__":
    # Load model
    tokenizer, model = load_model("./fake_news_model/huggingface_model")
    
    # Test examples
    examples = [
        "Scientists discover new treatment for cancer that shows promising results in clinical trials",
        "BREAKING: Famous celebrity secretly an alien, government officials confirm"
    ]
    
    for example in examples:
        prediction = predict(example, tokenizer, model)
        print(f"Text: {example}")
        print(f"Prediction: {prediction}\\n")
"""

with open(os.path.join(export_dir, "inference.py"), "w") as f:
    f.write(inference_script)

print(f"\nModel saved successfully in {export_dir}")
print(f"Total time elapsed: {(time.time() - start_time) / 60:.2f} minutes")