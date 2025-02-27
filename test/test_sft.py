import pytest
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig
from mol_gen_docking.sft_data import InstructionDatasetProcessor


@pytest.fixture(
    params=[
        "SMolInstruct",
    ],  # "Mol-Instructions"],
    scope="module",
)
def processor(request):
    name = request.param
    return InstructionDatasetProcessor(name, 8)


def test_instruction_dataset_processor(processor):
    """Test the InstructionDatasetProcessor with 1 process."""
    train, test = processor.get_training_corpus(100)
    assert "prompt" in train.column_names
    assert "completion" in train.column_names
    assert len(train) == 100
    assert len(test) == 10

    print(train[0])
    print(train[1])


def test_init_trainer(processor):
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")

    train, test = processor.get_training_corpus(100)
    training_args = SFTConfig(
        max_seq_length=1024,
        dataset_num_proc=8,
        packing=True,
    )
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train,
        eval_dataset=test,
        args=training_args,
    )

    print(trainer.train_dataset)
    print(tokenizer.decode(trainer.train_dataset[0]["input_ids"]))
