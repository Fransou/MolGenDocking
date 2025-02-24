import pytest
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer
from mol_gen_docking.sft_data import InstructionDatasetProcessor

@pytest.fixture(params=["SMolInstruct", "Mol-Instructions"])
def processor(name):
    return InstructionDatasetProcessor(name, 8)

def test_instruction_dataset_processor(name):
    """Test the InstructionDatasetProcessor with 1 process."""
    processor = InstructionDatasetProcessor(name, 8)
    train, test = processor.get_training_corpus()
    assert "prompt" in train.column_names
    assert "completion" in train.column_names

def test_init_trainer(name):
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")

    processor = InstructionDatasetProcessor(name, 8)
    train, test = processor.get_training_corpus()


    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train,
        eval_dataset=test,
    )