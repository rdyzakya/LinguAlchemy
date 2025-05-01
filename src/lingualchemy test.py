import argparse
import torch
import shutil
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification
)
from datasets import load_dataset, DatasetDict, concatenate_datasets
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from tqdm import tqdm
import pandas as pd
import os
import json


from src.fusion_models import *

os.environ['TOKENIZERS_PARALLELISM'] = 'true'




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="bert-base-multilingual-cased",
        help="Set the pre-trained model.",
    )  # "bigscience/mt0-base"
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="The epochs set for training.",
    )
    parser.add_argument(
        "--scale",
        type=int,
        default=10,
        help="The uriel scale set for training.",
    )
    parser.add_argument(
        "--vector",
        type=str,
        default=10,
        help="The uriel vector set for training.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Only train on the first 100 examples.",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default="/mnt/beegfs/farid/lingualchemy/ablation",
        help="Set the pre-trained model.",
    )
    parser.add_argument(
        "--eval_path",
        type=str,
        default="/home/alham.fikri/farid/lingualchemy/outputs",
        help="Set the evaluation dump file path.",
    )
    parser.add_argument(
        "--override_results",
        action="store_true",
        default=False,
        help="When enabled, remove the previous checkpoints results.",
    )
    parser.add_argument(
        "--wandb_offline", default=False, action="store_true", help="wandb offline mode"
    )
    
    args = parser.parse_args()

    if os.path.exists(args.out_path):
        assert (
            args.debug or args.override_results
        ), f"Output dir {args.out_path} already exists!"
        shutil.rmtree(args.out_path)
    os.makedirs(args.out_path)


    # if args.wandb_offline:
    #     os.environ["WANDB_MODE"] = "offline"
    os.environ["WANDB_MODE"] = "disabled"
    os.environ["WANDB_DISABLED"] = "true"

    complete_langs = ['id-ID', 'ta-IN', 'th-TH', 'tl-PH', 'en-US', 'km-KH', 'vi-VN', 'my-MM', 'jv-ID']

    def simplify_lang_code(lang):
        if lang == "zh-CN":
            return "zh_yue"
        elif lang == "zh-TW":
            return "zh"
        return lang.split("-")[0]

    simple_complete_langs = [simplify_lang_code(lang) for lang in complete_langs]

    dataset_test = {}
    for lang in tqdm(complete_langs):
        dataset_test[lang] = load_dataset("AmazonScience/massive", lang, split='test', trust_remote_code=True).remove_columns(
            ["id", "partition", "scenario", "annot_utt", "worker_id", "slot_method", "judgments"]
        )

    dset_test_dict = DatasetDict(dataset_test)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Load URIEL data
    current_script_path = os.path.abspath(__file__)
    current_script_base_path, _ = os.path.split(current_script_path)
    uriel_path = os.path.join(current_script_base_path, "..", "vectors", f"{args.vector}.pt")
    uriel_data = torch.load(uriel_path, weights_only=False)
    uriel_vector = torch.stack([torch.tensor(uriel_data[lang]) for lang in sorted(uriel_data.keys())])
    lang_to_index = {lang: idx for idx, lang in enumerate(sorted(uriel_data.keys()))}

    def encode_batch(batch):
        """Encodes a batch of input data using the model tokenizer."""
        lang_label = batch['locale'][0]
        encoding = tokenizer(batch["utt"], max_length=80, truncation=True, padding="max_length", return_tensors="pt")
        
        lang_index = lang_to_index[lang_label]
        encoding['language_labels'] = torch.tensor([lang_index] * len(batch['utt']))
        uriel_vec = uriel_vector[lang_index]
        encoding['uriel_labels'] = uriel_vec.repeat(len(batch['utt']), 1)
        
        return encoding

    if "intent" not in dset_test_dict.column_names:
        dset_test_dict = dset_test_dict.rename_column("intent", "labels")

    dset_test_dict = dset_test_dict.map(encode_batch, batched=True)

    # Initialize model
    config = AutoConfig.from_pretrained(args.model_name, num_labels=60)
    if args.model_name == "bert-base-multilingual-cased":
        print(args.model_name)
        model = FusionBertForSequenceClassification(config, uriel_vector)

        dset_test_dict.set_format(type="torch", columns=["labels", "utt", "input_ids", "token_type_ids", "attention_mask", "language_labels", "uriel_labels"])

    elif args.model_name == "xlm-roberta-base":
        print(args.model_name)
        model = FusionXLMRForSequenceClassification(config, uriel_vector)

        dset_test_dict.set_format(type="torch", columns=["labels", "utt", "input_ids", "attention_mask", "language_labels", "uriel_labels"])
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    model.eval()
    results = {}

    all_preds, all_labels = [], []

    for lang, dataset in dset_test_dict.items():
        # test_loader = DataLoader(dataset, batch_size=64)
        test_loader = DataLoader(dataset, batch_size=16)
        for batch in tqdm(test_loader, desc=f"Testing lang {lang}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            with torch.no_grad():
                logits, lang_logits, pooled_output = model(input_ids, attention_mask, None, None, None)[0]
                preds = torch.argmax(logits, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro')
        precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(all_labels, all_preds, average='micro')
        accuracy = accuracy_score(all_labels, all_preds)

        results[lang] = {
            'precision_macro': precision_macro,
            'precision_micro': precision_micro,
            'recall_macro': recall_macro,
            'recall_micro': recall_micro,
            'f1_macro': f1_macro,
            'f1_micro': f1_micro,
            'accuracy': accuracy
        }

    results_file_path = f"{args.eval_path}/{args.vector}_scores.json"
    print(results)
    if os.path.exists(results_file_path):
        if args.override_results:
            os.remove(results_file_path)
        else:
            raise AssertionError(f"Output file {results_file_path} already exists!")

    os.makedirs(f"{args.eval_path}", exist_ok=True)

    with open(
        results_file_path, "w", encoding="utf-8"
    ) as file:
        json.dump(results, file, indent=4)


