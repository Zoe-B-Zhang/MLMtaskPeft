# perft flan-t5

import json
import os
import shutil
import sys
from datetime import datetime

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (AutoModelForCausalLM, AutoModelForSeq2SeqLM,
                          AutoTokenizer, DataCollatorForLanguageModeling,
                          GemmaTokenizer, Seq2SeqTrainer,
                          Seq2SeqTrainingArguments, Trainer, TrainerCallback,
                          TrainerControl, TrainerState, TrainingArguments)

from .peft_prepare import prepare_dataset


def prepare_model_for_LoRA(my_model, my_lora_config):

    # model is not sharded, not use this step. could be in memory improvement in future.
    # prepare int-8 model for training
    # my_model = prepare_model_for_int8_training(my_model)

    # add LoRA adaptor
    my_model = get_peft_model(my_model, my_lora_config)
    trainable_params = my_model.print_trainable_parameters()
    return my_model, trainable_params


# for unsupervised learning, evalute the result by predict next token
#
def compute_perplexity(eval_pred):
    logits, labels = eval_pred
    # Shift the labels to the right to align them with the model's predictions
    # The model is trained to predict the next token in the sequence
    shift_labels = labels[..., 1:].contiguous().view(-1)
    shift_logits = logits[..., :-1, :].contiguous().view(-1, logits.size(-1))
    # Calculate loss
    loss_fct = torch.nn.CrossEntropyLoss(
        ignore_index=-100
    )  # Assuming -100 is used for ignored indices
    loss = loss_fct(shift_logits, shift_labels)

    # Perplexity is the exponential of the cross-entropy loss
    perplexity = torch.exp(loss).item()

    return {"perplexity": perplexity}


#
# make a CheckpointCallback,a subclass of TrainerCallback.
# It overrides the on_epoch_end method to save the model at the end of each epoch.
# The model.save_pretrained method is used to save the model, and the filename includes the epoch number.
#
# Custom callback to handle checkpoints
class MyCheckpointCallback(TrainerCallback):
    def on_train_begin(
        self, args, state: TrainerState, control: TrainerControl, **kwargs
    ) -> None:
        # Generate a timestamp when training begins
        state.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def on_evaluate(
        self, args, state: TrainerState, control: TrainerControl, metrics=None, **kwargs
    ) -> None:
        # Store the metrics in the state so they can be accessed in on_save
        if metrics is not None:
            state.evaluation_metrics = metrics

    def on_save(
        self, args, state: TrainerState, control: TrainerControl, **kwargs
    ) -> None:
        if control.should_save and hasattr(state, "evaluation_metrics"):
            output_dir = os.path.join(
                args.output_dir, state.timestamp, f"checkpoint-{state.global_step}"
            )
            os.makedirs(output_dir, exist_ok=True)
            eval_path = os.path.join(output_dir, "eval_results.json")
            with open(eval_path, "w") as f:
                json.dump(state.evaluation_metrics, f, indent=4)
            with open(os.path.join(output_dir, "README.md"), "w") as readme_file:
                readme_file.write(f"Checkpoint at global step {state.global_step}\n")


def peft_training(
    my_tokenizer,
    my_model,
    output_path,
    tokenized_train_dataset,
    tokenized_eval_dataset,
):
    # we want to ignore tokenizer pad token in the loss
    # label_pad_token_id = -100

    # Data collator
    """
    data_collator = DataCollatorForSeq2Seq(
        my_tokenizer,
        model=my_model,
        label_pad_token_id=label_pad_token_id,  # pad_to_multiple_of=8 forther perf improve
    )
    """
    data_collator = DataCollatorForLanguageModeling(my_tokenizer, mlm=False)

    # Define training args
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_path,
        per_device_train_batch_size=16,  # Adjust the batch size according to your hardwares
        learning_rate=1e-4,  # Learning rate might need to be fine-tuned for your specific task
        num_train_epochs=5,
        logging_dir=f"{output_path}/logs",
        logging_strategy="epoch",  # Log at the end of each epoch
        save_strategy="epoch",  # Save at the end of each epoch
        evaluation_strategy="epoch",  # Perform evaluation at the end of each epoch #eval_steps=500,  # Perform evaluation every X steps
        load_best_model_at_end=True,  # Load the best model based on the evaluation metric
        metric_for_best_model="loss",
        greater_is_better=False,  # For the loss, lower is better
    )

    trainer = Seq2SeqTrainer(
        model=my_model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_eval_dataset,
        tokenizer=my_tokenizer,
        callbacks=[
            MyCheckpointCallback()
        ],  # Replace with actual implementation of a checkpoint callback if needed
    )
    my_model.config.use_cache = (
        False  # silence the warnings. Please re-enable for inference!
    )
    trainer.train()

    # save the result.
    output_subfolder = getattr(
        trainer.state, "timestamp", datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    # Create the final output path by joining the base output path with the subfolder name
    final_output_path = os.path.join(output_path, output_subfolder)
    # Create the final output directory if it doesn't exist
    os.makedirs(final_output_path, exist_ok=True)

    # Save the model to the final output directory
    trainer.save_model(final_output_path)
    # Save the tokenizer to the output directory
    my_tokenizer.save_pretrained(final_output_path)

    # Save the training arguments
    with open(os.path.join(final_output_path, "training_args.json"), "w") as f:
        f.write(trainer.args.to_json_string())

    # Additionally, save the last evaluation results
    if hasattr(trainer.state, "evaluation_metrics"):
        eval_results_path = os.path.join(final_output_path, "eval_results.json")
        with open(eval_results_path, "w") as f:
            json.dump(trainer.state.evaluation_metrics, f, indent=4)

    return final_output_path


def get_directory_size(directory):
    total = 0.0
    for path, dirs, files in os.walk(directory):
        for f in files:
            fp = os.path.join(path, f)
            total += os.path.getsize(fp)
    total = total / 1024 / 1024 / 1024
    return total


def peft_task_on_seqModel(
    model_name, model_dir, token_json_file, data_json_file, if_w_mask
):
    output_path = os.getcwd() + "/data/peft"
    os.makedirs(output_path, exist_ok=True)
    output_pre_data_path = os.path.join(output_path, "data")
    os.makedirs(output_pre_data_path, exist_ok=True)

    my_tokenizer = AutoTokenizer.from_pretrained(model_dir)
    my_model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)

    with open(f"{output_pre_data_path}/before_{model_name}_model.txt", "w") as f:
        print(my_model, f)

    with open(f"{output_pre_data_path}/before_{model_name}_tokenizer.txt", "w") as f:
        print(my_tokenizer, f)

    tokenized_dataset, tokenized_eval, my_tokenizer, my_model = prepare_dataset(
        token_json_file,
        data_json_file,
        my_tokenizer,
        my_model,
        output_pre_data_path,
        if_w_mask,
    )

    # Define LoRA configuration
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q", "v"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    my_model, trainable_params = prepare_model_for_LoRA(
        my_model=my_model, my_lora_config=peft_config
    )

    final_output_path = peft_training(
        my_tokenizer=my_tokenizer,
        my_model=my_model,
        output_path=output_path,
        tokenized_train_dataset=tokenized_dataset,
        tokenized_eval_dataset=tokenized_eval,
    )

    # Move preprocessed data if the directories are different
    if output_pre_data_path != final_output_path:
        for file in os.listdir(output_pre_data_path):
            shutil.move(os.path.join(output_pre_data_path, file), final_output_path)

    directory_size = get_directory_size("final_output_path")

    return output_path, directory_size


def peft_task_on_llm(model_name, model_dir, token_json_file, data_json_file, if_w_mask):
    output_path = os.getcwd() + "/data/peft"
    print(f"output_path{output_path}")
    os.makedirs(output_path, exist_ok=True)
    output_pre_data_path = os.path.join(output_path, "data")
    os.makedirs(output_pre_data_path, exist_ok=True)

    print(model_dir)
    my_tokenizer = AutoTokenizer.from_pretrained(model_dir)
    # the following assign is from sft code of gemma examples. not sure the reason
    my_tokenizer.pad_token_id = my_tokenizer.eos_token_id
    my_tokenizer.pad_token = my_tokenizer.eos_token

    my_model = AutoModelForCausalLM.from_pretrained(
        model_dir, # device_map="auto", 
        attn_implementation="sdpa", # torch_dtype=torch.float16,
        device_map="cuda",
    )

    # my_model = my_model.to("cuda")

    # Enable gradient tracking for all model parameters
    # for param in my_model.parameters():
    #    param.requires_grad = True

    with open(f"{output_pre_data_path}/before_{model_name}_model.txt", "w") as f:
        print(my_model, f)

    with open(f"{output_pre_data_path}/before_{model_name}_tokenizer.txt", "w") as f:
        print(my_tokenizer, f)

    # rewrite prepare_dataset func
    # not use mask method, instead of next token prediction method
    tokenized_dataset, tokenized_eval, my_tokenizer, my_model = prepare_dataset(
        token_json_file,
        data_json_file,
        my_tokenizer,
        my_model,
        output_pre_data_path,
        if_w_mask,
    )

    # Define LoRA configuration
    modules = [
        "k_proj",
        "q_proj",
        "gate_proj",
        "o_proj",
        "v_proj",
        "down_proj",
        "up_proj",
    ]
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=modules,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    my_model, trainable_params = prepare_model_for_LoRA(
        my_model=my_model, my_lora_config=lora_config
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        my_tokenizer, mlm=False, mlm_probability=0.15
    )

    training_args = TrainingArguments(
        output_dir=output_path,
        per_device_train_batch_size=1,  # gradient_accumulation_steps=4,  #  optim="paged_adamw_32bit",
        save_strategy="epoch",  # Save at the end of each epoch
        logging_dir=f"{output_path}/logs",
        logging_strategy="epoch",  # Log at the end of each epoch
        learning_rate=2e-4,  # max_grad_norm=0.3,
        max_steps=100,
        warmup_ratio=0.03,  # fp16=True,
        # lr_scheduler_type="constant",
        # evaluation_strategy="epoch",  # Perform evaluation at the end of each epoch #eval_steps=500,  # Perform evaluation every X steps
        # load_best_model_at_end=True,  # Load the best model based on the evaluation metric
        # metric_for_best_model="loss",
        # greater_is_better=False,  # For the loss, lower is better
        # gradient_checkpointing=True,
    )

    # The tokenizer is involved in datacollecter
    trainer = Trainer(
        model=my_model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_dataset,
        eval_dataset=tokenized_eval,  # tokenizer=my_tokenizer,
        callbacks=[
            MyCheckpointCallback()
        ],  # Replace with actual implementation of a checkpoint callback if needed
    )

    """
    trainer = Trainer(
        model=my_model,
        train_dataset=tokenized_dataset,
        eval_dataset=tokenized_eval,
        args=TrainingArguments(
            per_device_train_batch_size=1, #gradient_accumulation_steps=4,
            warmup_steps=0.03,
            max_steps=100,
            learning_rate=2e-4,  # fp16=True,
            logging_steps=1,
            output_dir="outputs_mistral_b_finance_finetuned_test",
            optim="paged_adamw_8bit",
            save_strategy="epoch",
        ),
        data_collator=DataCollatorForLanguageModeling(my_tokenizer, mlm=False),
        callbacks=[MyCheckpointCallback()],
    )
    """

    my_model.config.use_cache = (
        False  # silence the warnings. Please re-enable for inference!
    )

    # Monitor GPU memory usage before training
    print("GPU memory usage before training:")
    print(torch.cuda.memory_summary())

    trainer.train()

    # Monitor GPU memory usage after training
    print("GPU memory usage after training:")
    print(torch.cuda.memory_summary())

    # save the result.
    output_subfolder = getattr(
        trainer.state, "timestamp", datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    # Create the final output path by joining the base output path with the subfolder name
    final_output_path = os.path.join(output_path, output_subfolder)
    # Create the final output directory if it doesn't exist
    os.makedirs(final_output_path, exist_ok=True)

    # Save the model to the final output directory
    trainer.save_model(final_output_path)
    # Save the tokenizer to the output directory
    my_tokenizer.save_pretrained(final_output_path)

    # Save the training arguments
    with open(os.path.join(final_output_path, "training_args.json"), "w") as f:
        f.write(trainer.args.to_json_string())

    # Additionally, save the last evaluation results
    if hasattr(trainer.state, "evaluation_metrics"):
        eval_results_path = os.path.join(final_output_path, "eval_results.json")
        with open(eval_results_path, "w") as f:
            json.dump(trainer.state.evaluation_metrics, f, indent=4)

    # Move preprocessed data if the directories are different
    if output_pre_data_path != final_output_path:
        for file in os.listdir(output_pre_data_path):
            shutil.move(os.path.join(output_pre_data_path, file), final_output_path)

    directory_size = get_directory_size("final_output_path")

    # Clear GPU memory cache
    torch.cuda.empty_cache()

    return output_path, directory_size


def peft_task(model_name, model_dir, token_json_file, data_json_file, if_w_mask):
    print(model_dir)
    if if_w_mask:
        output_path, directory_size = peft_task_on_llm(
            model_name, model_dir, token_json_file, data_json_file, if_w_mask
        )
    else:
        if model_name.lower() != "t5":
            print("without mask training data is only for t5 model ")

        output_path, directory_size = peft_task_on_seqModel(
            model_name, model_dir, token_json_file, data_json_file, if_w_mask
        )
    return output_path, directory_size


if __name__ == "__main__":
    print("test")
    token_json_file = "/workspace/data/add_tokens.json"

    if len(sys.argv) > 1:
        if sys.argv[1] == "1":
            data_json_file = "../peft/3gppterms_ft_w_masks_ft.json"
            model_dir = "../gemma-7b"
            model_name = "gemma"
            if_w_mask = True
        else:
            model_dir = "/workspace/model/google/flan-t5-base"
            data_json_file = "/workspace/data/ft_data.json"
            model_name = "t5"
            if_w_mask = False
        print(
            f"data_json_file:{data_json_file},token_json_file:{token_json_file},model_name:{model_name}, if_w_mask:{if_w_mask}"
        )

        output_path, directory_size = peft_task(
            model_name, model_dir, token_json_file, data_json_file, if_w_mask
        )
    else:
        print("give your choice pls ")
