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
- [OSS Component Usage Policy](#oss-component-usage-policy)
- [Code of Conduct](#code-of-conduct)
- [Where Can I Ask for Help?](#where-can-i-ask-for-help)

## Quick Links to Important Resources

- **Documentation**: [README.md](README.md) - Project overview and installation
- **Bug Reports**: [GitHub Issues](https://github.com/afifaniks/repo-qa/issues) - Report bugs and request features
- **License**: [LICENSE](LICENSE) - MIT License details
- **License Compliance**: [NOTICE](NOTICE) - Third-party attributions
- **OSS Policy**: [OSS Component Usage Policy](#oss-component-usage-policy) - Guidelines for dependencies

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
   
   # Check license compliance (if adding dependencies)
   make license-check
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
   - License compliance confirmation (for dependency changes)
   - Checklist completion

### Dependencies and Licensing

If your changes add, remove, or modify dependencies, please review our [OSS Component Usage Policy](#oss-component-usage-policy) and ensure:
- All new dependencies use compatible licenses
- License checker passes: `make license-check`
- NOTICE file is updated if needed: `make generate-notice-direct`

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

## OSS Component Usage Policy

To ensure license compatibility and maintain project quality, contributors must follow these guidelines when adding or modifying dependencies:

### License Compatibility Requirements

#### Allowed Licenses

We allow permissive licenses. Example:
- **MIT License** - Permissive, compatible with our project
- **Apache-2.0** - Permissive with patent grants
- **BSD (2-Clause, 3-Clause)** - Simple permissive licenses

#### Prohibited Licenses

We do not comply with prohibitive copyleft licenses. Example:
- **GPL (v2, v3)** - Strong copyleft that could restrict distribution
- **AGPL** - Network copyleft incompatible with commercial use
- **LGPL** - Weak copyleft requiring dynamic linking considerations
- **CC-BY-SA** - Share-alike requirements incompatible with MIT
- **Proprietary/Commercial** - Licensing restrictions and costs

### Dependency Requirements

**Source Requirements:**
- **Use trusted sources only**: PyPI, conda-forge, or well-established repositories
- **Avoid unofficial forks** unless absolutely necessary and pre-approved
- **Check maintenance status**: Active development, recent releases, responsive maintainers
- **Security considerations**: No known vulnerabilities, good security track record

**Before Adding Dependencies:**

1. **Check if it's really needed** - Can the functionality be implemented internally?
2. **Evaluate alternatives** - Are there lighter-weight or better-maintained options?
3. **Check the dependency tree** - What transitive dependencies does it bring?
4. **Verify license compatibility** - Use our license checker: `make license-check`

### Approval Process

**For New Dependencies:**

1. **Open an issue first** to discuss the need and proposed dependency
2. **Include in your PR description:**
   - Why this dependency is needed
   - License information and compatibility assessment
   - Alternative options considered
   - Dependency size and impact analysis
   - Link to the package's repository and documentation

3. **Run license checks:**
   ```bash
   # After adding the dependency
   make license-check
   
   # Generate updated NOTICE file
   make generate-notice-direct
   ```

4. **Maintainer review** will verify:
   - License compatibility with project goals
   - Security and maintenance considerations
   - Impact on project size and complexity
   - Alignment with project architecture

### License Checking Tools

We provide automated tools to help maintain compliance:

```bash
# Check all licenses and compatibility
make license-check

# Generate detailed license report  
make license-report

# Generate JSON format for CI/CD
make license-json

# Update NOTICE file with attributions
make generate-notice-direct
```

### Common License Scenarios

**✅ Safe to add:**
```bash
# MIT licensed package
pip install some-mit-package

# Apache-2.0 with patent grants
pip install apache-licensed-tool

# BSD-style permissive license
pip install bsd-utility
```

**⚠️ Requires discussion:**
```bash
# Multiple licenses (need to verify)
pip install dual-licensed-package

# Custom license (needs manual review)
pip install custom-license-tool
```

**❌ Not allowed:**
```bash
# GPL licensed (copyleft)
pip install gpl-package

# Proprietary with restrictions
pip install commercial-only-tool
```

### Handling License Issues

**If you're unsure about a license:**
1. **Create an issue** with the package name and license details
2. **Don't include it in your PR** until approved
3. **Provide context** about why you need this specific package
4. **Suggest alternatives** if you know of any

**If our license checker flags an issue:**
1. **Don't ignore the warning** - it's there for a reason
2. **Check if it's a false positive** (sometimes license metadata is incorrect)
3. **Look for alternative packages** with compatible licenses
4. **Discuss with maintainers** if you believe it should be allowed

### Contributing to License Policy

This policy evolves with the project. If you have suggestions for improvements:

- **Open an issue** to discuss policy changes
- **Provide examples** of packages that should be allowed/prohibited
- **Share experiences** with license compatibility in your projects
- **Help improve our tooling** for automated license checking

### Educational Resources

**Learn more about open source licenses:**
- [Choose a License](https://choosealicense.com/) - License comparison tool
- [SPDX License List](https://spdx.org/licenses/) - Standard license identifiers
- [GitHub License Guide](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/licensing-a-repository) - Repository licensing
- [OSI Approved Licenses](https://opensource.org/licenses) - Open Source Initiative approved licenses

**Questions about licensing?** Feel free to ask in [GitHub Discussions](https://github.com/afifaniks/repo-qa/discussions) or email afifaniks@gmail.com.

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
