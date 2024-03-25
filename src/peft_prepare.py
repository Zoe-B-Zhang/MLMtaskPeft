"""
Preparing the new tokens and corresonding training data for fine-tuning tokenizer
    1. put all abbrs in 3GPP_21905-h10-abbr.csv to a text file
    2. check whether those abbrs are in current tokenizers' vocabulary or not and add checking result as a new column in 3GPP_21905-h10-abbr.csv
    3. fine-tuning tokenizer
        1. use abbrs those are not in vocabulary as new tokens
        2. use the abbr's explanation as training data
And new additial tokens to tokenizer
"""
import json
import os
import random
import sys

import pandas as pd
import torch

# from datasets import load_dataset
from datasets import Dataset
from torch.utils.data import DataLoader

# from torch.utils.data import Dataset  # DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
)

# 1. create the finetuning tokenizer data used


# Precondition:
# handing csv file with columns of "Full Name", 'Abbreviation','invoc'
# There are seperate steps to get csv file of abbr from 3GPP_21905-h10 standard documents.
# All the abbrs in csv file will be checked in vocabulary of flan-t5 tokenizer. the checking
# result will be saved in new invoc column.
# the final file will be '3GPP_21905-h10-abbr-invoc.csv'

company_string = "Nokia or NOKIA stands for Nokia Corporation, natively known as Nokia Oyj in Finnish and Nokia Abp in Swedish, is a Finnish multinational corporation that specializes in telecommunications, information technology, and consumer electronics1.Nokia has a rich history and has operated in various industries over the past 150 years. It was founded as a pulp mill and had long been associated with rubber and cables1. Today, Nokia is a technology leader across mobile, fixed, and cloud networks, enabling a more productive, sustainable, and inclusive world."


def get_additional_tokens(csv_file):
    # Read the CSV file
    df = pd.read_csv(csv_file)

    # Select the column you want to extract
    column = df["Abbreviation"]
    data_column = df["Full Name"]

    # Select the column to check for 0 values
    check_column = df["invoc"]

    # Write the contents of the column to a list when the check column is 0
    token_list = [item for i, item in enumerate(column) if check_column[i] == 0]

    data_list = [
        f"{item1} stands for {item2}"
        for i, (item1, item2) in enumerate(zip(column, data_column))
        if check_column[i] == 0
    ]

    data_list = [company_string] + data_list

    return token_list, data_list


def save_list_to_json(token_list, token_file_name, data_list, data_file_name):
    """
    This function saves two lists to two separate json files.

    Parameters:
    token_list (list): The list to be saved in the token file.
    token_file_name (str): The name of the token file.
    data_list (list): The list to be saved in the data file.
    data_file_name (str): The name of the data file.
    """

    # Convert the list to a list of dictionaries
    train_data = [{"text": s} for s in data_list]

    # Write the data to a JSON Lines file
    with open(data_file_name, "w") as f:
        for item in train_data:
            f.write(json.dumps(item) + "\n")

    # Save the token_list to a json file
    with open(token_file_name, "w") as file:
        json.dump(token_list, file)


def read_list_from_json(token_file_name, data_file_name):
    """
    This function reads two lists from two separate json files.

    Parameters:
    token_file_name (str): The name of the token file.
    data_file_name (str): The name of the data file.

    Returns:
    tuple: A tuple containing two lists read from the json files.
    """

    # Read the data from a JSON Lines file
    with open(data_file_name, "r") as f:
        data_list = [json.loads(line) for line in f]

    # Read the token_list from a json file
    with open(token_file_name, "r") as file:
        token_list = json.load(file)

    return token_list, data_list


def read_tokens_from_json(token_file_name):
    # Read the token_list from a json file
    with open(token_file_name, "r") as file:
        token_list = json.load(file)

    return token_list


# 2. Add new tokens in Tokenizer
def add_tokens(my_tokenizer, my_model, additional_token_list):
    # Add special tokens
    special_tokens_dict = {"additional_special_tokens": additional_token_list}
    my_tokenizer.add_special_tokens(special_tokens_dict)
    my_model.resize_token_embeddings(len(my_tokenizer))
    return my_tokenizer, my_model


# 3 preparing dataset
class DatasetPreparation:
    def __init__(self):  # , tokenizer):
        # self.tokenizer = tokenizer
        print("init")

    # Create mask randomly to ask model generate text so to evaluate it
    # Input: "The quick <extra_id_0> over the lazy dog"
    # Target: "The quick brown fox jumps over the lazy dog"
    # try <extra_id_0> as mask in text based on mask usage :https://github.com/huggingface/transformers/issues/21211
    def create_pretext_task(self, text):
        words = text.split()
        start_infill = random.randint(0, len(words) - 2)
        end_infill = start_infill + random.randint(1, 2)
        infilled_words = words[:start_infill] + ["<extra_id_0>"] + words[end_infill:]
        infilled_text = " ".join(infilled_words)
        return infilled_text, text

    # create temp json file for training
    def preprocess_func(self, examples, save_path):
        inputs, targets = [], []
        pretext_results = []
        for doc in examples["text"]:
            infilled_text, original_text = self.create_pretext_task(doc)
            inputs.append(infilled_text)
            targets.append(original_text)
            pretext_results.append({"input": infilled_text, "target": original_text})
        # Save pretext results to a JSON file
        with open(save_path, "w") as f:
            json.dump(pretext_results, f)

    def preprocess_function(self, my_tokenizer, examples, save_path):
        inputs, targets = [], []
        pretext_results = []
        for doc in examples["text"]:
            infilled_text, original_text = self.create_pretext_task(doc)
            inputs.append(infilled_text)
            targets.append(original_text)
            pretext_results.append({"input": infilled_text, "target": original_text})

        # Save pretext results to a JSON file
        with open(save_path, "w") as f:
            json.dump(pretext_results, f)

        model_inputs = my_tokenizer(
            inputs, max_length=None, truncation=True, padding="max_length"
        )
        with my_tokenizer.as_target_tokenizer():
            labels = my_tokenizer(
                targets, max_length=None, truncation=True, padding="max_length"
            )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs


class MaskDatasetPreparation:
    def __init__(self):
        print("Initialization of MaskDatasetPreparation")

    def preprocess_function(self, tokenizer, examples):
        # Note: Assumption is that 'examples' is a batch from the dataset
        # Tokenization of the inputs and targets
        model_inputs = tokenizer(
            examples["input"], truncation=True, max_length=1024, padding="max_length"
        )
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                examples["target"],
                truncation=True,
                max_length=1024,
                padding="max_length",
            )

        model_inputs["labels"] = labels["input_ids"]

        return model_inputs

    def gemma_preprocess_function(self, tokenizer, examples):
        # Note: Assumption is that 'examples' is a batch from the dataset
        # Tokenization of the inputs and targets
        model_inputs = tokenizer(
            examples["input"], truncation=True, max_length=1024, padding=True
        )
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                examples["target"],
                truncation=True,
                max_length=1024,
                padding="max_length",
            )

        input_ids = model_inputs["input_ids"]
        labels = labels["input_ids"]

        return dict(
            input_ids=torch.tensor(input_ids),
            labels=torch.tensor(labels),
        )


#  preparing dataset and split it into train and test(used as eval) for unsupervised learning
#  prepare_mask_dataset is for new preprocessed json data with masked result.
#  prepare_orig_dataset is for old json data, will random generate the single mask
def prepare_mask_dataset(
    token_json_file, data_json_file, my_tokenizer, my_model, output_dir
):
    token_list = read_tokens_from_json(token_json_file)
    # add tokens in tokenizer
    my_tokenizer, my_model = add_tokens(
        my_tokenizer=my_tokenizer, my_model=my_model, additional_token_list=token_list
    )

    # Load the JSON file into a Python object
    with open(data_json_file, "r") as f:
        data = json.load(f)

    # Create the Hugging Face Dataset from the list of dictionaries
    full_dataset = Dataset.from_dict(
        {
            "input": [dic["input"] for dic in data],
            "target": [dic["target"] for dic in data],
        }
    )

    # Split the data into 80% train, 20% test
    train_test_split = full_dataset.train_test_split(test_size=0.2)

    # A train and test (evaluation) dataset
    train_dataset = train_test_split["train"]
    eval_dataset = train_test_split["test"]

    dataset_preparation = MaskDatasetPreparation()

    # Training data tokenization
    tokenized_traindata = train_dataset.map(
        lambda examples: dataset_preparation.gemma_preprocess_function(
            my_tokenizer, examples
        ),
        batched=True,
        remove_columns=[
            "input",
            "target",
        ],  # Remove the original columns to keep only the tokenized data
    )

    # Save tokenized data to a JSON file for the training set

    # Ensure the output directory exists, create it if necessary
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    tokenized_traindata.save_to_disk(f"{output_dir}/train_tokenized")
    # with open(f"{output_dir}/train_tokenized.json", "w", encoding='utf-8') as f:
    # json.dump(tokenized_traindata, f, ensure_ascii=False, indent=4)

    # Evaluation data tokenization
    tokenized_eval = eval_dataset.map(
        lambda examples: dataset_preparation.gemma_preprocess_function(
            my_tokenizer, examples
        ),
        batched=True,
        remove_columns=[
            "input",
            "target",
        ],  #  Remove the original columns to keep only the tokenized data
    )

    tokenized_eval.save_to_disk(f"{output_dir}/eval_tokenized")

    # Save tokenized data to a JSON file for the evaluation set
    # with open(f"{output_dir}/eval_tokenized.json", "w", encoding='utf-8') as f:
    # json.dump(tokenized_eval, f, ensure_ascii=False, indent=4)

    print(tokenized_traindata)
    print(tokenized_eval)
    print(my_tokenizer)

    return (
        tokenized_traindata,
        tokenized_eval,
        my_tokenizer,
        my_model,
    )


def prepare_orig_dataset(
    token_json_file, data_json_file, my_tokenizer, my_model, output_dir
):
    token_list = read_tokens_from_json(token_json_file)
    # add tokens in tokenizer
    my_tokenizer, my_model = add_tokens(
        my_tokenizer=my_tokenizer, my_model=my_model, additional_token_list=token_list
    )

    # get training data

    # Load the datasets
    # Load the full dataset (no predefined splits)
    # full_dataset = load_dataset("json", data_files=data_json_file)
    # Read the JSON file into a Python object
    data = []
    with open(data_json_file, "r") as f:
        for line in f:
            data.append(json.loads(line))

    # Convert the Python list of dictionaries to a Hugging Face Dataset
    # full_dataset = Dataset.from_dict({"data": data})

    # Convert the Python list of dictionaries to a Hugging Face Dataset
    data_columns = {key: [dic[key] for dic in data] for key in data[0]}

    # Create the Hugging Face Dataset from the dictionary of columns
    full_dataset = Dataset.from_dict(data_columns)
    # split the data into 80% train, 20% test.
    train_test_split = full_dataset.train_test_split(test_size=0.2)

    # a train and test (evaluation) dataset
    train_dataset = train_test_split["train"]
    eval_dataset = train_test_split["test"]

    # Initialize DatasetPreparation class
    # there are two kinds of data json file
    # old one: {"text": text training data}
    # new one which preprocessed to get masks by entropy value
    dataset_preparation = DatasetPreparation()

    # training data
    tokenized_traindata = train_dataset.map(
        lambda examples: dataset_preparation.preprocess_function(
            my_tokenizer, examples, f"{output_dir}/train_pretext.json"
        ),
        batched=True,
        remove_columns=["text"],
    )

    tokenized_traindata.save_to_disk(f"{output_dir}/train_tokenized")
    print(f"Keys of tokenized train dataset: {list(tokenized_traindata.features)}")

    # evaluation data
    tokenized_eval = eval_dataset.map(
        lambda examples: dataset_preparation.preprocess_function(
            my_tokenizer, examples, f"{output_dir}/eval_pretext.json"
        ),
        batched=True,
        remove_columns=["text"],
    )
    tokenized_eval.save_to_disk(f"{output_dir}/eval_tokenized")
    print(f"Keys of tokenized eval dataset: {list(tokenized_eval.features)}")

    return (
        tokenized_traindata,
        tokenized_eval,
        my_tokenizer,
        my_model,
    )


def prepare_dataset(
    token_json_file, data_json_file, my_tokenizer, my_model, output_dir, with_mask
):
    if with_mask:
        (
            tokenized_traindata,
            tokenized_eval,
            my_tokenizer,
            my_model,
        ) = prepare_mask_dataset(
            token_json_file, data_json_file, my_tokenizer, my_model, output_dir
        )
    else:
        (
            tokenized_traindata,
            tokenized_eval,
            my_tokenizer,
            my_model,
        ) = prepare_orig_dataset(
            token_json_file, data_json_file, my_tokenizer, my_model, output_dir
        )

    return (
        tokenized_traindata,
        tokenized_eval,
        my_tokenizer,
        my_model,
    )


def my_collate_fn(batch):
    input_ids = torch.stack([torch.tensor(item["input_ids"]) for item in batch])
    attention_mask = torch.stack(
        [torch.tensor(item["attention_mask"]) for item in batch]
    )
    labels = torch.stack([torch.tensor(item["labels"]) for item in batch])
    return input_ids, attention_mask, labels


if __name__ == "__main__":
   print(test)
