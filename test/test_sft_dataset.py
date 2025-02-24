import pytest
from mol_gen_docking.sft_data import InstructionDatasetProcessor


@pytest.mark.parametrize("name", ["Mol-Instructions", "SMolInstruct"])
def test_instruction_dataset_processor(name:str):
    """Test the InstructionDatasetProcessor with 1 process."""
    processor = InstructionDatasetProcessor(name,8)
    train, test = processor.get_training_corpus(100, 10)
    assert len(train) == 100
    assert len(test) == 10



