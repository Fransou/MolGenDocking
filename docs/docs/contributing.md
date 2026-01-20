# Contributing to MolGenDocking

Thank you for your interest in contributing to MolGenDocking! This guide explains how to contribute code, datasets, and documentation.

## Code of Conduct

We are committed to providing a welcoming and inclusive environment for all contributors. Please treat all community members with respect and follow these principles:

- Be respectful and constructive
- Welcome diverse perspectives
- Give credit to others
- Report inappropriate behavior to maintainers

## Getting Started

### 1. Fork and Clone

```bash
# Fork on GitHub (click Fork button)

# Clone your fork
git clone https://github.com/YOUR_USERNAME/MolGenDocking.git
cd MolGenDocking

# Add upstream remote
git remote add upstream https://github.com/Fransou/MolGenDocking.git
```

### 2. Create Development Branch

```bash
# Update from upstream
git fetch upstream
git checkout upstream/main

# Create feature branch
git checkout -b feature/your-feature-name

# Naming conventions:
# feature/add-new-property
# bugfix/fix-docking-crash
# docs/improve-api-docs
```

### 3. Set Up Development Environment

```bash
# Install in editable mode with dev dependencies
pip install -e ".[model]"

# Install linting/formatting tools
pip install black flake8 isort mypy pytest

# Install pre-commit hooks
pip install pre-commit
pre-commit install
```

## Development Workflow

### Making Changes

**Code Style**:
- Follow PEP 8
- Use type hints
- Maximum line length: 100 characters

```python
# Good example
def compute_qed(smiles: str) -> Optional[float]:
    """
    Compute QED (Quantitative Estimate of Drug-likeness).

    Args:
        smiles: SMILES string of molecule

    Returns:
        QED score (0-1) or None if invalid
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return compute_score(mol)
```

**Format Code**:
```bash
# Auto-format with black
black mol_gen_docking/ test/

# Sort imports
isort mol_gen_docking/ test/

# Check style
flake8 mol_gen_docking/ test/

# Type checking
mypy mol_gen_docking/
```

### Testing

**Write Tests**:
```python
# test/test_new_feature.py
import pytest
from mol_gen_docking.feature import MyFunction

def test_basic_functionality():
    result = MyFunction(input_data)
    assert result == expected_output

def test_edge_case():
    with pytest.raises(ValueError):
        MyFunction(invalid_input)

@pytest.mark.slow
def test_docking_computation():
    # Slower tests marked with @pytest.mark.slow
    pass
```

**Run Tests**:
```bash
# Run all tests
pytest test/

# Run specific test file
pytest test/test_new_feature.py -v

# Run with coverage
pytest test/ --cov=mol_gen_docking

# Run only fast tests
pytest test/ -m "not slow"
```

### Documentation

**Update docstrings**:
```python
def process_molecules(
    smiles_list: List[str],
    target: str = "GSK3B",
    verbose: bool = False
) -> List[Dict[str, float]]:
    """
    Process list of molecules and compute properties.

    This function evaluates molecular structures against a specific
    protein target using docking and other scoring methods.

    Args:
        smiles_list: List of SMILES strings to process
        target: Target protein name (default: "GSK3B")
        verbose: Print progress information (default: False)

    Returns:
        List of dictionaries containing:
        - smiles: Input SMILES
        - validity: Boolean validity flag
        - docking_score: Binding affinity prediction
        - properties: Additional molecular properties

    Raises:
        ValueError: If target protein not found
        RuntimeError: If docking computation fails

    Examples:
        >>> results = process_molecules(['CCO', 'CC(C)O'])
        >>> print(results[0]['docking_score'])
        -7.5
    """
```

**Update documentation files**:
- Edit `.md` files in `docs/docs/`
- Preview: `mkdocs serve`
- Build: `mkdocs build`

### Commit and Push

```bash
# Review changes
git status
git diff

# Stage changes
git add mol_gen_docking/
git add test/
git add docs/

# Commit with descriptive message
git commit -m "feat: add new scoring function for XYZ property

- Implement XyzScorer class with GPU acceleration
- Add unit tests with 95% coverage
- Update documentation with examples"

# Push to your fork
git push origin feature/your-feature-name
```

**Commit Message Guidelines**:
```
<type>: <short summary (50 chars max)>

<longer description (wrap at 72 chars)>
- Bullet point for major changes
- Another important detail

Fixes #123
Closes #456
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

### Create Pull Request

1. Visit GitHub and create PR from your fork to upstream
2. Fill out PR template:
   ```markdown
   ## Description
   Brief description of changes

   ## Type of Change
   - [ ] Bug fix
   - [ ] New feature
   - [ ] Documentation update

   ## Related Issues
   Fixes #123

   ## Testing
   How to test these changes

   ## Checklist
   - [ ] Tests pass locally
   - [ ] Code follows style guide
   - [ ] Documentation updated
   - [ ] No breaking changes
   ```

3. Wait for CI checks and reviews

## Contributing Types

### Bug Reports

**File an issue** on GitHub:

```markdown
## Description
Brief description of the bug

## Steps to Reproduce
1. Run this code...
2. With these parameters...
3. Then observe...

## Expected Behavior
What should happen

## Actual Behavior
What actually happens

## Environment
- OS: Ubuntu 22.04
- Python: 3.10.5
- GPU: RTX 3090
- AutoDock-GPU: 1.5.3

## Additional Context
```

### Feature Requests

```markdown
## Description
Description of desired feature

## Motivation
Why this feature is needed

## Proposed Solution
How it could be implemented

## Alternatives
Other approaches considered
```

### Documentation

Edit documentation files directly:

```bash
# Documentation lives in docs/docs/
docs/
├── index.md                 # Homepage
├── datasets/
│   ├── overview.md
│   ├── docking.md
│   ├── generation.md
│   └── properties.md
├── benchmarks/
│   ├── overview.md
│   └── metrics.md
├── reward_server/
│   ├── getting_started.md
│   ├── configuration.md
│   └── api.md
└── installation.md
```

### Dataset Contributions

Contributing new datasets or data processing scripts:

1. **Prepare Dataset**:
   - Validate SMILES/SMARTS
   - Compute molecular properties
   - Organize in standard format
   - Create metadata file

2. **Document**:
   - Create `data/YOUR_DATASET/README.md`
   - Describe source, format, size
   - Include usage examples

3. **Submit**:
   - Create GitHub issue proposing dataset
   - Discuss integration approach
   - Prepare data upload

### Model Contributions

Contributing pre-trained models or baseline implementations:

1. **Implement Model**:
   ```python
   # mol_gen_docking/models/your_model.py
   class YourModel(GenerativeModel):
       def __init__(self, config):
           ...

       def generate(self, prompt: str, **kwargs) -> str:
           ...
   ```

2. **Add Configuration**:
   ```yaml
   # configs/models/your_model.yaml
   model:
     name: your_model
     version: 1.0
     checkpoint_path: models/your_model.pt
   ```

3. **Benchmarking**:
   - Run on standard benchmarks
   - Report metrics and reproducibility
   - Document training procedure

## Review Process

### Code Review Guidelines

When reviewing, consider:

✅ **Approve if**:
- Code follows style guide
- Tests are comprehensive
- Documentation is clear
- No performance regressions
- Changes are focused

🔄 **Request Changes if**:
- Missing test coverage
- Unclear variable names
- Documentation gaps
- Performance concerns
- Breaking changes without discussion

❌ **Close if**:
- Off-topic
- Low quality
- No response to feedback

### Responding to Feedback

```bash
# Make requested changes
git add .
git commit -m "refactor: address review feedback"
git push origin feature/your-feature-name

# Don't force-push (maintains discussion history)
# Maintainers will squash commits when merging
```

## Release Procedure

Maintainers follow semantic versioning:

- **MAJOR**: Breaking changes (e.g., 1.0.0 → 2.0.0)
- **MINOR**: New features (e.g., 1.0.0 → 1.1.0)
- **PATCH**: Bug fixes (e.g., 1.0.0 → 1.0.1)

## Project Structure

```
MolGenDocking/
├── mol_gen_docking/          # Main package
│   ├── models/               # Generative models
│   ├── reward/               # Scoring functions
│   ├── evaluation/           # Evaluation metrics
│   ├── data/                 # Data loading/processing
│   └── server.py             # FastAPI server
├── test/                     # Unit tests
├── docs/                     # Documentation
├── notebooks/               # Example notebooks
├── data/                    # Datasets (not in repo)
└── slurm/                   # HPC job scripts
```

## Performance Considerations

When contributing code:

1. **GPU Memory**: Consider memory usage on 12GB GPUs
2. **Throughput**: Benchmark against baselines
3. **Caching**: Implement where appropriate
4. **Async Operations**: Use async/await for I/O

```python
# Good: Efficient memory usage
def batch_process(molecules: List[str], batch_size: int = 32):
    for i in range(0, len(molecules), batch_size):
        batch = molecules[i:i+batch_size]
        yield process_batch(batch)

# Avoid: Loading everything into memory
results = [score_molecule(m) for m in huge_molecule_list]
```

## Documentation Standards

### API Documentation

```python
def compute_property(
    smiles: str,
    property_name: str,
    **kwargs
) -> float:
    """
    Compute molecular property.

    [1-2 sentence summary]

    [Longer description if needed]

    Parameters
    ----------
    smiles : str
        SMILES string of molecule
    property_name : str
        Name of property to compute
    **kwargs : dict
        Additional arguments passed to scorer

    Returns
    -------
    float
        Computed property value

    Raises
    ------
    ValueError
        If SMILES is invalid

    See Also
    --------
    validate_smiles : Validate SMILES syntax

    Examples
    --------
    >>> compute_property('CCO', 'qed')
    0.85
    """
```

## Communication

- **Issues**: For bugs and feature requests
- **Discussions**: For questions and ideas
- **Pull Requests**: For code contributions
- **Email**: Contact maintainers directly for sensitive topics

## Recognition

Contributors are recognized in:
- `CONTRIBUTORS.md` file
- Release notes for significant contributions
- GitHub contributors page

## License

By contributing, you agree that your contributions are licensed under the project's license.

## Questions?

- Check existing issues and discussions
- Read the documentation
- Open a new discussion if needed

Thank you for contributing! 🚀
