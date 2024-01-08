from config import (
    datasets_max_in_memory,
    dataset,
    code,
    cache_dir,
    pretrained,
    language,
    batch_size,
    output_dir,
    evaluate_metric
)

################################################################################
# Setup Config
################################################################################

import datasets

datasets.config.IN_MEMORY_MAX_SIZE = 1024 * 1024 * 1024 * datasets_max_in_memory

################################################################################
# Download Datasets
################################################################################

from datasets import load_dataset, DatasetDict

common_voice = DatasetDict()

common_voice["train"] = load_dataset(
    dataset, code, split="train+validation", use_auth_token=True, cache_dir=cache_dir
)
common_voice["test"] = load_dataset(
    dataset, code, split="test", use_auth_token=True, cache_dir=cache_dir
)

common_voice = common_voice.remove_columns(
    [
        "accent",
        "age",
        "client_id",
        "down_votes",
        "gender",
        "locale",
        "path",
        "segment",
        "up_votes",
    ]
)

print(common_voice)

################################################################################
# Download Whisper Components
################################################################################

from transformers import (
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperProcessor,
    WhisperForConditionalGeneration,
)

feature_extractor = WhisperFeatureExtractor.from_pretrained(pretrained)
tokenizer = WhisperTokenizer.from_pretrained(
    pretrained, language=language, task="transcribe"
)
processor = WhisperProcessor.from_pretrained(pretrained, task="transcribe")
model = WhisperForConditionalGeneration.from_pretrained(pretrained)

model.config.forced_decoder_ids = None
model.config.suppress_tokens = []

################################################################################
# Extract Data Features
################################################################################

from datasets import Audio
import multiprocessing

common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000))


def prepare_dataset(batch):
    # load and resample audio data from 48 to 16kHz
    audio = batch["audio"]

    # compute log-Mel input features from input audio array
    batch["input_features"] = feature_extractor(
        audio["array"], sampling_rate=audio["sampling_rate"]
    ).input_features[0]

    # encode target text to label ids
    batch["labels"] = tokenizer(batch["sentence"]).input_ids
    return batch


common_voice = common_voice.map(
    prepare_dataset,
    remove_columns=common_voice.column_names["train"],
    cache_file_names={
        "train": f"{cache_dir}/common_voice_train_processed.arrow",
        "test": f"{cache_dir}/common_voice_test_processed.arrow",
    },
)

################################################################################
# Build Training Data
################################################################################

import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [
            {"input_features": feature["input_features"]} for feature in features
        ]
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

################################################################################
# Setup Evaluation
################################################################################

import evaluate

metric = evaluate.load(evaluate_metric)


def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    result = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {evaluate_metric: result}


################################################################################
# Setup Training Config
################################################################################

from transformers import Seq2SeqTrainingArguments

training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=1,
    learning_rate=1e-5,
    warmup_steps=500,
    max_steps=4000,
    gradient_checkpointing=True,
    fp16=True,
    evaluation_strategy="steps",
    per_device_eval_batch_size=batch_size // 2,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=1000,
    eval_steps=1000,
    logging_steps=25,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model=evaluate_metric,
    greater_is_better=False,
    push_to_hub=True,
)

################################################################################
# Fine Tune
################################################################################

from transformers import Seq2SeqTrainer

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=common_voice["train"],
    eval_dataset=common_voice["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)

processor.save_pretrained(training_args.output_dir)
trainer.train()

################################################################################
# Upload
################################################################################

kwargs = {
    "dataset_tags": dataset,
    "dataset": "Common Voice 11.0",
    "dataset_args": f"config: {code}, split: test",
    "language": code,
    "model_name": f"Whisper Small {code} - {language}",
    "finetuned_from": pretrained,
    "tasks": "automatic-speech-recognition",
    "tags": "hf-asr-leaderboard",
}

trainer.push_to_hub(**kwargs)
