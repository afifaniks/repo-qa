# RepoQA: Repository-level Question Answering with RAG

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

![](assets/preview.jpg)
<small><i>Image generated with ChatGPT</i></small>

RepoQA is a software **repository-level question answering system** powered by **Retrieval-Augmented Generation (RAG)** and a **Large Language Model (LLM)**.  
It allows users to ask natural language questions about a software repository and receive context-aware answers grounded in the repository’s code and documentation.

## Features (Tentative)
- **Repository-level Question Answering**: Ask questions about functions, classes, modules, or overall repository design.
- **Retrieval-Augmented Generation (RAG)**: Combines semantic retrieval with generative reasoning for precise and grounded answers.
- **Code-Aware Retrieval**: Extractsrelevant files, functions, and comments from the repository to provide evidence-based responses.
- **LLM Integration**: Uses a large language model to generate human-like, context-rich answers.

## Architecture
1. **Indexing**:  
   - The repository codebase is parsed and embedded using a code-aware embedding model.  
   - Metadata (file paths, function definitions, docstrings) is stored in a vector database.

2. **Query Processing**:  
   - User provides a natural language query.  
   - The system retrieves the most relevant code snippets/files using semantic similarity search.  

3. **Answer Generation**:  
   - The retrieved context is passed into an LLM with the user’s question.  
   - The LLM generates a grounded and contextually accurate answer.  

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details on:

- How to set up the development environment
- Our coding standards and style guide
- How to submit changes and report bugs
- Where to ask for help

### Quick Start for Contributors

```bash
# Clone your fork
git clone https://github.com/<your-username>/repo-qa.git
cd repo-qa

# Set up development environment
make setup

# Run tests to verify everything works
make test

# See all available commands
make help
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### OSS Component Usage Policy

All dependencies are checked for license compatibility. See our:
- [NOTICE](NOTICE) file for third-party attributions  
- [OSS Component Usage Policy](CONTRIBUTING.md#oss-component-usage-policy) for contribution guidelines
- [License Policy Quick Reference](docs/license-policy.md) for allowed/prohibited licenses

## Generative AI Usage
GitHub Copilot with Claude Sonnet 4 was used to generate boilerplates, documentation, etc. for this project. However, all the generated contents were manually validated.