from dotenv import load_dotenv
import os
from sqlalchemy import create_engine, text as sql_text
import pandas as pd
from tqdm.auto import tqdm
from transformers import TrainingArguments, Trainer, EarlyStoppingCallback
import transformers as tfm
import numpy as np
import evaluate
import logging
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from datasets import Dataset, Features, Value, ClassLabel, DatasetDict
from sklearn.model_selection import train_test_split

# Get the root logger
logger = logging.getLogger()

# Set the logging level
logger.setLevel(logging.INFO)

# Create handlers for both console and file
console_handler = logging.StreamHandler()
file_handler = logging.FileHandler("output.log")

# Create a formatter
formatter = logging.Formatter(
    "%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S"
)

# Set the formatter for both handlers
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Add both handlers to the logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)

load_dotenv()


logger.info("Connecting to database")
conn = create_engine(
    f"postgresql://{os.getenv('PG_USER')}:{os.getenv('PG_PASSWORD')}@{os.getenv('PG_HOST')}/{os.getenv('PG_DBNAME')}"
).connect()

logger.info("Reading data from database")
temp_readings = pd.read_sql_query(sql_text("SELECT * FROM public.temp_readings"), conn)
fire_alerts = pd.read_sql_query(sql_text("SELECT * FROM public.fire_alerts"), conn)
tweets = pd.read_sql_query(sql_text("SELECT * FROM public.tweets"), conn)

# output all tweets that occur at the date and location of a fire event
# first, get all the dates and locations of fire events
logger.info("Processing fire events and tweets")
fires = []

for i, row in fire_alerts.iterrows():
    fires.append((row["event_day"], row["xy"]))

# then, get all the tweets that occur at the date and location of a fire event
tqdm.pandas()

fire_tweets = tweets.progress_apply(
    lambda row: (row["day"], row["xy"]) in fires, axis=1
)

fire_samples = tweets.loc[fire_tweets]
not_fire_samples = tweets.loc[~fire_tweets]

tweet_samples_x = (
    fire_samples["content"].values.tolist()
    + not_fire_samples["content"].values.tolist()
)
tweet_samples_y = [1] * len(fire_samples) + [0] * len(not_fire_samples)

# preprocess our data
# we need to split our data into train and test sets first
# then we need to correct any class imbalance
logger.info("Preprocessing data and splitting into train, test, and validation sets")
train_x, test_x, train_y, test_y = train_test_split(
    tweet_samples_x, tweet_samples_y, test_size=0.2, stratify=tweet_samples_y
)
train_x, val_x, train_y, val_y = train_test_split(
    train_x, train_y, test_size=0.2, stratify=train_y
)


features = Features(
    {
        "text": Value("string"),
        "label": ClassLabel(num_classes=2, names=["no_fire", "yes_fire"]),
    }
)

dataset = DatasetDict(
    {
        "train": Dataset.from_dict(
            {"text": train_x, "label": train_y}, features=features
        ),
        "test": Dataset.from_dict({"text": test_x, "label": test_y}, features=features),
        "validation": Dataset.from_dict(
            {"text": val_x, "label": val_y}, features=features
        ),
    }
)

logger.info("Tokenizing datasets")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")


def tokenize_function(examples):
    return tokenizer(
        examples["text"], padding="max_length", truncation=True, max_length=300
    )


tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns("text")


id2label = {0: "no_fire", 1: "yes_fire"}
label2id = {"no_fire": 0, "yes_fire": 1}

logger.info("Initializing model")

model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased", num_labels=2, id2label=id2label, label2id=label2id
)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


tfm.logging.set_verbosity_info()

logger.info("Setting up training arguments")
training_args = TrainingArguments(
    output_dir="test_trainer",
    evaluation_strategy="steps",
    save_strategy="steps",
    num_train_epochs=3,
    load_best_model_at_end=True,
    auto_find_batch_size=True,
    report_to="none",
    per_device_train_batch_size=128,
    per_device_eval_batch_size=128,
)
metric = evaluate.load("accuracy")

logger.info("Creating Trainer and beginning training")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"].shuffle(seed=42),
    eval_dataset=tokenized_datasets["validation"].shuffle(seed=42),
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)

trainer.train()

# evaluate the model
logger.info("Evaluating the model")
trainer.evaluate(tokenized_datasets["test"])

logger.info("Saving the model")
trainer.save_model()
