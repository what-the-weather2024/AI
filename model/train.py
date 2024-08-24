import evaluate
import numpy as np
import torch
from datasets import ClassLabel, DatasetDict, load_dataset
from transformers import (
    Trainer,
    TrainingArguments,
    ViTFeatureExtractor,
    ViTForImageClassification,
)

merged_data_path = "/Users/imdohun/PycharmProjects/wtw/AI/data/merged_data"
dataset_path = f"{merged_data_path}/data"
dataset = load_dataset(dataset_path)

label_names = sorted(set(dataset["train"]["label"]))
print("lables: ", label_names)
dataset = dataset.cast_column("label", ClassLabel(names=label_names))

dataset = dataset.shuffle(seed=42)

train_split = dataset["train"].train_test_split(train_size=0.8)

ds = DatasetDict({"train": train_split["train"], "test": train_split["test"]})

print("Training Dataset")
print(ds["train"])
print(ds["train"][0])
print(ds["train"][-1])

print("Testing Dataset")
print(ds["test"])
print(ds["test"][0])
print(ds["test"][-1])


MODEL_CKPT = "google/vit-base-patch16-224-in21k"
MODEL_NAME = MODEL_CKPT + "-weather_images_classification"
NUM_OF_EPOCHS = 1

LEARNING_RATE = 2e-4
STEPS = 100

BATCH_SIZE = 16
DEVICE = torch.device("mps")

# REPORTS_TO = 'tensorboard'

feature_extractor = ViTFeatureExtractor.from_pretrained(MODEL_CKPT)


def transform(sample_batch):
    # take a list of PIL images and turn them into pixel values
    inputs = feature_extractor(
        [x.convert("RGB") for x in sample_batch["image"]], return_tensors="pt"
    )

    # prepare labels
    inputs["labels"] = sample_batch["label"]
    return inputs


prepped_ds = ds.with_transform(transform)


def data_collator(batch):
    return {
        "pixel_values": torch.stack([x["pixel_values"] for x in batch]),
        "labels": torch.tensor([x["labels"] for x in batch]),
    }


def compute_metrics(p):
    accuracy_metric = evaluate.load("accuracy")
    accuracy = accuracy_metric.compute(
        predictions=np.argmax(p.predictions, axis=1), references=p.label_ids
    )["accuracy"]

    ### ------------------- F1 scores -------------------

    f1_score_metric = evaluate.load("f1")
    weighted_f1_score = f1_score_metric.compute(
        predictions=np.argmax(p.predictions, axis=1),
        references=p.label_ids,
        average="weighted",
    )["f1"]
    micro_f1_score = f1_score_metric.compute(
        predictions=np.argmax(p.predictions, axis=1),
        references=p.label_ids,
        average="micro",
    )["f1"]
    macro_f1_score = f1_score_metric.compute(
        predictions=np.argmax(p.predictions, axis=1),
        references=p.label_ids,
        average="macro",
    )["f1"]

    ### ------------------- recall -------------------

    recall_metric = evaluate.load("recall")
    weighted_recall = recall_metric.compute(
        predictions=np.argmax(p.predictions, axis=1),
        references=p.label_ids,
        average="weighted",
    )["recall"]
    micro_recall = recall_metric.compute(
        predictions=np.argmax(p.predictions, axis=1),
        references=p.label_ids,
        average="micro",
    )["recall"]
    macro_recall = recall_metric.compute(
        predictions=np.argmax(p.predictions, axis=1),
        references=p.label_ids,
        average="macro",
    )["recall"]

    ### ------------------- precision -------------------

    precision_metric = evaluate.load("precision")
    weighted_precision = precision_metric.compute(
        predictions=np.argmax(p.predictions, axis=1),
        references=p.label_ids,
        average="weighted",
    )["precision"]
    micro_precision = precision_metric.compute(
        predictions=np.argmax(p.predictions, axis=1),
        references=p.label_ids,
        average="micro",
    )["precision"]
    macro_precision = precision_metric.compute(
        predictions=np.argmax(p.predictions, axis=1),
        references=p.label_ids,
        average="macro",
    )["precision"]

    return {
        "accuracy": accuracy,
        "Weighted F1": weighted_f1_score,
        "Micro F1": micro_f1_score,
        "Macro F1": macro_f1_score,
        "Weighted Recall": weighted_recall,
        "Micro Recall": micro_recall,
        "Macro Recall": macro_recall,
        "Weighted Precision": weighted_precision,
        "Micro Precision": micro_precision,
        "Macro Precision": macro_precision,
    }


labels = dataset["train"].features["label"].names

model = ViTForImageClassification.from_pretrained(
    MODEL_CKPT,
    num_labels=len(labels),
    id2label={str(i): c for i, c in enumerate(labels)},
    label2id={c: str(i) for i, c in enumerate(labels)},
).to(DEVICE)

args = TrainingArguments(
    output_dir=MODEL_NAME,
    remove_unused_columns=False,
    num_train_epochs=NUM_OF_EPOCHS,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    disable_tqdm=False,
    load_best_model_at_end=True,
    metric_for_best_model="Weighted F1",
    logging_first_step=True,
    # hub_private_repo=True,
    # push_to_hub=True
)

trainer = Trainer(
    model=model,
    args=args,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    train_dataset=prepped_ds["train"],
    eval_dataset=prepped_ds["test"],
    tokenizer=feature_extractor,
)

train_results = trainer.train()

trainer.save_model()
trainer.log_metrics("train", train_results.metrics)
trainer.save_metrics("train", train_results.metrics)
trainer.save_state()


metrics = trainer.evaluate(prepped_ds["test"])
trainer.log_metrics("eval", metrics)
trainer.save_metrics("eval", metrics)
