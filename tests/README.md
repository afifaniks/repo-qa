# Testing Guide for RepoQA

This directory contains comprehensive tests for the RepoQA project.

## Test Structure

The test directory mirrors the source code structure for easy navigation:

```
tests/
├── __init__.py              # Package initialization
├── conftest.py              # Shared fixtures and configuration
├── test_config.py           # Configuration module tests
├── test_app.py              # Main application tests
├── test_api.py              # API endpoint tests
├── embedding/               # Tests for embedding module
│   ├── __init__.py
│   └── test_sentence_transformer.py
├── indexing/                # Tests for indexing module
│   ├── __init__.py
│   └── test_git_indexer.py
├── llm/                     # Tests for LLM module
│   ├── __init__.py
│   └── test_llm_factory.py
└── pipeline/                # Tests for pipeline module
    ├── __init__.py
    ├── test_rag.py          # RAG pipeline tests
    └── test_agentic_rag.py  # Agentic RAG pipeline tests
```

This structure mirrors the `repoqa/` source directory, making it easy to find tests for any module.

## Running Tests

### Install Test Dependencies

First, ensure you have all test dependencies installed:

```bash
pip install -e ".[dev]"
```

Or install directly from requirements:

```bash
pip install pytest pytest-cov pytest-mock black isort flake8 mypy
```

### Run All Tests

```bash
pytest
```

### Run Tests with Coverage

```bash
pytest --cov=repoqa --cov-report=html --cov-report=term-missing
```

This will generate:
- Terminal output showing coverage percentages
- HTML coverage report in `htmlcov/index.html`
- XML coverage report in `coverage.xml`

### Run Specific Test Files

```bash
# Test configuration module
pytest tests/test_config.py

# Test embedding module
pytest tests/embedding/

# Test indexing module
pytest tests/indexing/

# Test LLM factory
pytest tests/llm/

# Test pipeline modules
pytest tests/pipeline/

# Test API endpoints
pytest tests/test_api.py

# Test a specific module
pytest tests/embedding/test_sentence_transformer.py
pytest tests/pipeline/test_rag.py
```

### Run Specific Test Classes or Methods

```bash
# Run a specific test class
pytest tests/test_config.py::TestConfig

# Run a specific test method
pytest tests/test_config.py::TestConfig::test_config_loads_defaults
```

### Run Tests with Different Markers

```bash
# Run only unit tests
pytest -m unit

# Run only integration tests
pytest -m integration

# Skip slow tests
pytest -m "not slow"
```

### Run Tests with Verbose Output

```bash
pytest -v
```

### Run Tests and Stop on First Failure

```bash
pytest -x
```

## Test Coverage

The test suite covers:

### Core Modules

**Configuration (`test_config.py`)**
- ✅ Loading default configuration
- ✅ Overriding with user configuration
- ✅ Environment variable overrides
- ✅ Deep merging of nested configs
- ✅ Type conversion (bool, int, float)
- ✅ Property accessors

**Main Application (`test_app.py`)**
- ✅ RepoQA initialization in RAG mode
- ✅ RepoQA initialization in agent mode
- ✅ Invalid mode error handling
- ✅ Repository indexing
- ✅ Question answering

**API Endpoints (`test_api.py`)**
- ✅ Root endpoint
- ✅ Health check endpoint
- ✅ Ask endpoint with new repository
- ✅ Ask endpoint with existing repository
- ✅ Force update functionality
- ✅ Error handling
- ✅ Input validation
- ✅ Collection management functions

### Embedding Module (`embedding/`)

**Sentence Transformer (`test_sentence_transformer.py`)**
- ✅ Model initialization with device selection
- ✅ Encoding single text
- ✅ Encoding multiple texts
- ✅ Batch encoding with custom batch size
- ✅ Getting embedding dimensions
- ✅ Passing custom kwargs

### Indexing Module (`indexing/`)

**Git Indexer (`test_git_indexer.py`)**
- ✅ File chunking and processing
- ✅ Finding code files in repositories
- ✅ Ignore pattern matching
- ✅ Git URL detection
- ✅ Cloning repositories
- ✅ Handling encoding errors
- ✅ Extracting git metadata

### LLM Module (`llm/`)

**LLM Factory (`test_llm_factory.py`)**
- ✅ Getting existing models
- ✅ Pulling missing models
- ✅ Connection error handling
- ✅ Pull failure handling
- ✅ Custom kwargs support
- ✅ Backend validation

### Pipeline Module (`pipeline/`)

**RAG Pipeline (`test_rag.py`)**
- ✅ Pipeline initialization
- ✅ Safe document retrieval
- ✅ Document formatting
- ✅ Response cleaning
- ✅ Query processing
- ✅ Error handling
- ✅ Source file tracking

**Agentic RAG Pipeline (`test_agentic_rag.py`)**
- ✅ Agent initialization
- ✅ Semantic search tool
- ✅ Similarity search with scores
- ✅ Directory listing tool
- ✅ File reading tool
- ✅ Agent execution
- ✅ Error handling
- ✅ File access tracking

## Continuous Integration

The test suite is designed to run in CI/CD environments. Example GitHub Actions workflow:

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'
      - name: Install dependencies
        run: |
          pip install -e ".[dev]"
      - name: Run tests
        run: |
          pytest --cov=repoqa --cov-report=xml
      - name: Upload coverage
        uses: codecov/codecov-action@v3
```

## Writing New Tests

### Test Naming Conventions

- Test files: `test_<module>.py`
- Test classes: `Test<ClassName>`
- Test methods: `test_<what_it_tests>`

### Using Fixtures

Fixtures are defined in `conftest.py` and are automatically available in all tests:

```python
def test_example(temp_dir, mock_embedding_model):
    # temp_dir is a temporary directory
    # mock_embedding_model is a mocked embedding model
    pass
```

### Mocking External Dependencies

Use `unittest.mock` or `pytest-mock` to mock external dependencies:

```python
from unittest.mock import Mock, patch

@patch('repoqa.module.external_dependency')
def test_with_mock(mock_dependency):
    mock_dependency.return_value = "mocked value"
    # Your test code
```

## Code Quality

### Run Linters

```bash
# Black (code formatting)
black tests/

# isort (import sorting)
isort tests/

# flake8 (linting)
flake8 tests/

# mypy (type checking)
mypy tests/
```

### Pre-commit Checks

Consider setting up pre-commit hooks:

```bash
pip install pre-commit
pre-commit install
```

## Troubleshooting

### Tests Fail Due to Missing Dependencies

Ensure all dependencies are installed:
```bash
pip install -e ".[dev]"
```

### Tests Fail Due to Environment Variables

Some tests check environment variable behavior. Ensure your environment is clean:
```bash
unset LLM_MODEL
pytest tests/test_config.py
```

### Import Errors

Make sure the package is installed in development mode:
```bash
pip install -e .
```

## Coverage Goals

- **Overall coverage**: Aim for >85%
- **Critical modules**: >90% coverage
  - config.py
  - pipeline modules
  - indexing modules
- **API endpoints**: 100% coverage

## Contributing

When adding new features:
1. Write tests first (TDD approach recommended)
2. Ensure all tests pass
3. Maintain or improve coverage
4. Update this README if needed

## Resources

- [pytest documentation](https://docs.pytest.org/)
- [pytest-cov documentation](https://pytest-cov.readthedocs.io/)
- [unittest.mock documentation](https://docs.python.org/3/library/unittest.mock.html)
