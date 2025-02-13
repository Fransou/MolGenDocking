from datasets import load_dataset, Dataset
import math
from transformers import DataCollatorForLanguageModeling, TrainingArguments, AutoModelForCausalLM, AutoTokenizer, Trainer
import torch

dataset = load_dataset("zjunlp/Mol-Instructions", 'Molecule-oriented Instructions', trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained("crumb/nano-mistral")
tokenizer = AutoTokenizer.from_pretrained("crumb/nano-mistral")

def get_training_corpus(dataset):
    corpus = []
    for k in dataset.keys():
        for i in range(len(dataset[k])):
            line = "<instr>" + dataset[k][i]["instruction"] + dataset[k][i]["input"] + "<\instr>" + " " + dataset[k][i]["output"]
            corpus.append(line)
            print(line)
            if i>100:
                break
    return corpus


corpus = get_training_corpus(dataset)
tokenizer.add_special_tokens({"additional_special_tokens": ["<instr>", "<\instr>"], "mask_token": "[MASK]"})
# tokenizer.train_new_from_iterator(corpus, vocab_size=tokenizer.vocab_size+100)

dataset = Dataset.from_dict({"text": corpus})
def tokenize_function(examples):
    result = tokenizer(examples["text"])
    if tokenizer.is_fast:
        result["word_ids"] = [result.word_ids(i) for i in range(len(result["input_ids"]))]
    return result

lm_dataset = dataset.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"])

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

train_size = 100
test_size = int(0.1 * train_size)

downsampled_dataset = lm_dataset.train_test_split(
    train_size=train_size, test_size=test_size, seed=42
)


batch_size = 64
# Show the training loss with every epoch
logging_steps = len(downsampled_dataset["train"]) // batch_size

training_args = TrainingArguments(
    output_dir=f"test",
    overwrite_output_dir=True,
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    push_to_hub=False,
    fp16=True,
    logging_steps=logging_steps,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=downsampled_dataset["train"],
    eval_dataset=downsampled_dataset["test"],
    data_collator=data_collator,
    tokenizer=tokenizer,
)



eval_results = trainer.evaluate()
print(f">>> Perplexity: {math.exp(eval_results['eval_loss']):.2f}")

trainer.train()


eval_results = trainer.evaluate()
print(f">>> Perplexity: {math.exp(eval_results['eval_loss']):.2f}")
