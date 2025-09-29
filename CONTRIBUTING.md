# Contributing to VLM Run FiftyOne Plugin

Thank you for your interest in contributing to the VLM Run FiftyOne plugin! This document provides guidelines and information for contributors.

## Getting Started

1. Fork the repository
2. Clone your fork locally
3. Create a new branch for your feature or bugfix
4. Make your changes
5. Add tests for your changes
6. Ensure all tests pass
7. Submit a pull request

## Development Setup

1. Install the plugin in development mode:
```bash
pip install -e .
```

2. Install development dependencies:
```bash
pip install -r requirements-dev.txt
```

3. Run tests:
```bash
python -m pytest tests/
```

## Code Style

- Follow PEP 8 style guidelines
- Use type hints where appropriate
- Add docstrings to all functions and classes
- Keep functions focused and small
- Use meaningful variable and function names

## Testing

- Write tests for all new functionality
- Ensure existing tests continue to pass
- Use mocking for external API calls
- Test error conditions and edge cases

## Documentation

- Update README.md for new features
- Add docstrings to all new functions
- Include usage examples
- Update the plugin manifest (fiftyone.yml) if needed

## Pull Request Process

1. Ensure your code follows the style guidelines
2. Add tests for your changes
3. Update documentation as needed
4. Ensure all tests pass
5. Submit a pull request with a clear description

## Reporting Issues

When reporting issues, please include:

- Plugin version
- FiftyOne version
- Python version
- Operating system
- Steps to reproduce
- Expected vs actual behavior
- Error messages (if any)

## Feature Requests

For feature requests, please:

- Describe the feature clearly
- Explain the use case
- Consider implementation complexity
- Check if similar functionality already exists

## License

By contributing to this project, you agree that your contributions will be licensed under the MIT License.

## Questions?

If you have questions about contributing, please open an issue or contact the maintainers.
