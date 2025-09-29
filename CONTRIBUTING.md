# Contributing to RepoQA

Thank you for your interest in RepoQA! 

We welcome and encourage all kinds of contributions to Repo-QA! This project aims to build intelligent repository level question-answering using AI and vector databases.

## Table of Contents

- [Quick Links to Important Resources](#quick-links-to-important-resources)
- [Testing](#testing)
- [Environment Setup](#environment-setup)
- [How to Submit Changes](#how-to-submit-changes)
- [How to Report a Bug](#how-to-report-a-bug)
- [How to Request an Enhancement](#how-to-request-an-enhancement)
- [Style Guide & Coding Conventions](#style-guide--coding-conventions)
- [Code of Conduct](#code-of-conduct)
- [Where Can I Ask for Help?](#where-can-i-ask-for-help)

## Quick Links to Important Resources

- **Documentation**: [README.md](README.md) - Project overview and installation
- **Bug Reports**: [GitHub Issues](https://github.com/afifaniks/repo-qa/issues) - Report bugs and request features
- **License**: [LICENSE](LICENSE) - MIT License details
- **License Compliance**: [NOTICE](NOTICE) - Third-party attributions

## Testing

Our test suite ensures code quality and functionality. Here's how to work with tests:

### Running Tests

```bash
[TBD]
```

### Test Structure

- **Unit tests**: Located in `tests/` directory
- **Test framework**: We use `pytest`
- **Coverage**: We aim for >80% test coverage
- **Test markers**: Use `@pytest.mark.slow` for slow tests, `@pytest.mark.integration` for integration tests

## Environment Setup

### Prerequisites

- Python 3.10 or higher
- Virtual environment (recommended)

### Setup Steps

1. **Fork and clone the repository**:
   ```bash
   git clone https://github.com/<your-username>/repo-qa.git
   cd repo-qa
   ```

2. **Create and activate virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   make install
   # or manually:
   pip install -e ".[dev]"
   ```

4. **Verify setup**:
   ```bash
   make test
   python -m repoqa.license_checker --help
   ```

## How to Submit Changes

We follow the GitHub Flow for contributions:

### Pull Request Process

1. **Create a feature branch** from `main`:
   ```bash
   git checkout -b feature/amazing-new-feature
   ```

2. **Make your changes** and ensure they follow our standards:
   ```bash
   # Format code
   make format
   
   # Run linting
   make lint
   
   # Run tests
   make test
   ```

3. **Commit with clear messages** following conventional commits:
   ```bash
   git commit -m "Add SPDX license identifier mapping"
   git commit -m "Update contributing guidelines"
   ```

4. **Push to your fork** and create a Pull Request:
   ```bash
   git push origin feature/amazing-new-feature
   ```

5. **Fill out the PR template** with:
   - Clear description of changes
   - Link to related issues
   - Screenshots/examples if applicable
   - Checklist completion

## How to Report a Bug 🐞🪲🐛

Bugs are problems in code functionality or... or... "unxepected" behavior. We appreciate bug reports!

### Before Submitting a Bug Report

1. [**Search existing issues**](https://github.com/afifaniks/repo-qa/issues) to avoid duplicates
2. **Try the latest version** to ensure the bug still exists
3. **Gather information** about your environment

### Bug Report Template

Use this template when creating a bug report:

```markdown
**Bug Description**
A clear and concise description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
1. Go to '...'
2. Run command '...'
3. See error

**Expected Behavior**
What you expected to happen.

**Actual Behavior**
What actually happened.

**Environment**
- OS: [e.g., macOS 12.0, Ubuntu 20.04, Windows 11]
- Python version: [e.g., 3.9.0]
- Repo-QA version: [e.g., 1.0.0]
- Virtual environment: [yes/no]

**Additional Context**
- Error messages or logs
- Screenshots if applicable
- Any other relevant information
```

### Good First Issues

Looking for your first contribution? Look for issues labeled:
- `good first issue` - Perfect for newcomers
- `help wanted` - We'd love community help on these
- `documentation` - Improve our docs
- `bug` with `low priority` - Good learning opportunities

## How to Request an Enhancement

Enhancements are new features or improvements to existing functionality.

### Before Submitting an Enhancement

1. **Check existing issues** for similar requests
2. **Consider the scope** - is this aligned with project goals?
3. **Think about implementation** - do you have ideas for how it could work?

### Enhancement Request Template

```markdown
**Enhancement Description**
A clear and concise description of the enhancement.

**Problem It Solves**
What problem does this enhancement solve? Is it related to frustration with existing functionality?

**Proposed Solution**
Describe the solution you'd like to see.

**Alternative Solutions**
Describe any alternative solutions you've considered.

**Additional Context**
- Screenshots, mockups, or examples
- Links to related issues or discussions
- Any other relevant context
```

## Style Guide & Coding Conventions

### Python Code Style

- **Follow [PEP 8](https://peps.python.org/pep-0008/)** with these specifics.

### Code Formatting

We use automated formatting tools:

```bash
# Format all code
make format

# Or manually:
black repoqa/
isort repoqa/
```

### Type Hints

- **Use type hints** for function parameters and return values
- **Import types** from `typing` module when needed
- **Example**:
  ```python
  from typing import Dict, List, Optional
  
  def check_licenses(dependencies: List[str]) -> Dict[str, str]:
      """Check license compatibility for dependencies."""
      pass
  ```

### Documentation

- **Docstrings**: Use Google-style docstrings
- **Comments**: Explain "why" not "what"
- **README updates**: Update documentation for new features

### Example Docstring

```python
def map_to_spdx(license_name: str) -> Optional[str]:
    """Map a license name to its SPDX identifier.
    
    Args:
        license_name: The license name to map (e.g., "MIT License")
        
    Returns:
        The SPDX identifier if found, None otherwise.
        
    Examples:
        >>> map_to_spdx("MIT License")
        "MIT"
        >>> map_to_spdx("Unknown License")
        None
    """
```

### Branch Name Format

We maintain the following branch naming convention:

- New Feature: feature/add_feature_x
- Task: task/add_task_y
- BugFix: bugfix/fix_bug_that_doesnt_work
- Patch: patch/patch_not_working_release

## Code of Conduct

### Our Pledge

We are committed to providing a welcoming and inclusive experience for everyone, regardless of:
- Age, body size, disability, ethnicity, gender identity and expression
- Level of experience, education, socio-economic status
- Nationality, personal appearance, race, religion
- Sexual identity and orientation

### Our Standards

**Positive behaviors**:
- Using welcoming and inclusive language
- Being respectful of differing viewpoints and experiences
- Gracefully accepting constructive criticism
- Focusing on what is best for the community
- Showing empathy towards other community members

**Unacceptable behaviors**:
- Harassment, trolling, or discriminatory language
- Publishing others' private information without permission
- Professional misconduct or inappropriate behavior
- Any conduct that would be inappropriate in a professional setting

### Enforcement

Project maintainers are responsible for clarifying standards and will take appropriate action in response to unacceptable behavior.

**Contact**: For Code of Conduct concerns, email afif.mamun@ucalgary.ca


## Where Can I Ask for Help?

### Getting Help

Don't hesitate to ask for help! Here are the best places:

1. **GitHub Issues** - [Bug reports and feature requests](https://github.com/afifaniks/repo-qa/issues)
   - Bug reports and technical issues
   - Feature requests and enhancements
   - Documentation improvements

3. **Direct Contact** - afifaniks@gmail.com
   - Private or sensitive matters
   - Collaboration opportunities
   - Code of Conduct concerns

### Best Practices for Getting Help

- **Search first**: Check existing issues and discussions
- **Be specific**: Provide context, error messages, and environment details
- **Be patient**: We're volunteers with day jobs, but we care about helping
- **Be kind**: A little politeness goes a long way
- **Follow up**: Let us know if suggested solutions work

### Contributing Questions

Have questions about contributing? Here are common questions:

- **"I'm new to open source, where do I start?"** → Look for `good first issue` labels
- **"How do I set up the development environment?"** → Follow our [Environment Setup](#environment-setup) guide
- **"I found a typo, do I need to create an issue?"** → Small fixes can go directly to a PR
- **"My PR was rejected, what now?"** → Don't worry! Ask for feedback and try again

## License

By contributing to Repo-QA, you agree that your contributions will be licensed under the [MIT License](LICENSE).

**Thank you for contributing to Repo-QA!**

Every contribution, no matter how small, makes this project better. We appreciate your time, effort, and creativity in helping build something amazing together.
