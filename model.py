import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch

# Load the dataset
df_org = pd.read_csv(r"D:\All_data_science_project\NLP\finetune_1\7allV03.csv")

# Prepare labels
labels = df_org['category'].unique().tolist()
labels = [s.strip() for s in labels]
NUM_LABELS = len(labels)

id2label = {id: label for id, label in enumerate(labels)}
label2id = {label: id for id, label in enumerate(labels)}

# Assign numeric labels
df_org['labels'] = pd.factorize(df_org['category'])[0]

# Initialize the tokenizer and model
tokenizer = BertTokenizer.from_pretrained('dbmdz/bert-base-turkish-uncased', max_length=512)
model = BertForSequenceClassification.from_pretrained('dbmdz/bert-base-turkish-uncased', num_labels=NUM_LABELS, id2label=id2label, label2id=label2id)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Ensure all texts are strings
df_org['text'] = df_org['text'].astype(str)

# Split the data
SIZE = df_org.shape[0]
train_texts = list(df_org['text'][:SIZE//2])
val_texts = list(df_org['text'][SIZE//2:(3*SIZE)//4])
test_texts = list(df_org['text'][(3*SIZE)//4:])
train_labels = list(df_org['labels'][:SIZE//2])
val_labels = list(df_org['labels'][SIZE//2:(3*SIZE)//4])
test_labels = list(df_org['labels'][(3*SIZE)//4:])

# Tokenize the data
train_encodings = tokenizer(train_texts, truncation=True, padding=True)
val_encodings = tokenizer(val_texts, truncation=True, padding=True)
test_encodings = tokenizer(test_texts, truncation=True, padding=True)

# Verify the sizes of the splits
print(f'Train size: {len(train_texts)}, Validation size: {len(val_texts)}, Test size: {len(test_texts)}')

# Define a custom dataset class
class GPReviewDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

# Create datasets
train_dataset = GPReviewDataset(train_encodings, train_labels)
val_dataset = GPReviewDataset(val_encodings, val_labels)
test_dataset = GPReviewDataset(test_encodings, test_labels)

# Define compute_metrics function (dummy implementation, you should define it based on your needs)
def compute_metrics(p):
    return {"accuracy": (p.predictions.argmax(-1) == p.label_ids).mean()}

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',          # output directory
    do_train=True,
    do_eval=True,
    num_train_epochs=1,              # total number of training epochs
    per_device_train_batch_size=16,  # batch size per device during training
    per_device_eval_batch_size=32,
    warmup_steps=100,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_strategy='steps',
    logging_dir='logs',
    logging_steps=50,
    eval_strategy='steps',
    eval_steps=50,
    save_strategy='steps',
    #fp16=True,
    load_best_model_at_end=True,
)

# Initialize the Trainer
trainer = Trainer(
    model=model,                         # the instantiated  Transformers model
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=val_dataset,            # evaluation dataset
    compute_metrics=compute_metrics
)

# Train the model
trainer.train()

# Save the model and tokenizer
save_directory = "./saved_models"
model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)

print("Model and tokenizer saved successfully.")