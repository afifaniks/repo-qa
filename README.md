# RepoQA: Repository-le3. **Answer Generation**:  
   - The retrieved context is passed into an LLM with the user's question.  
   - The LLM generates a grounded and contextually accurate answer.  

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Copyright © 2025 Afif Al Mamun Question Answering with RAG

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