# Contributing to DynamicCompactDetect

Thank you for considering contributing to DynamicCompactDetect!

## Code of Conduct

This project is maintained by Abhilash Chadhar and Divya Athya. We expect all contributors to adhere to the [code of conduct](CODE_OF_CONDUCT.md). Please read it before participating.

## How to Contribute

1. **Fork the repository**: Create your own fork of the repository on GitHub.

2. **Clone your fork**: 
   ```
   git clone https://github.com/your-username/dynamiccompactdetect.git
   cd dynamiccompactdetect
   ```

3. **Create a branch**: Create a new branch for your contribution.
   ```
   git checkout -b feature/your-feature-name
   ```

4. **Make your changes**: Implement your feature or bug fix.

5. **Add tests**: Add tests for your changes to ensure they work correctly.

6. **Run tests**: Make sure all tests pass.
   ```
   python -m tests.run_all_tests
   ```

7. **Format your code**: Ensure your code follows the project's style guidelines.
   ```
   black .
   isort .
   ```

8. **Commit your changes**: Commit your changes with a clear commit message.
   ```
   git add .
   git commit -m "Add feature: your feature description"
   ```

9. **Push to your fork**: Push your changes to your fork on GitHub.
   ```
   git push origin feature/your-feature-name
   ```

10. **Create a Pull Request**: Go to the original repository on GitHub and create a pull request from your fork.

## Pull Request Process

1. Update the README.md or documentation with details of changes if appropriate.
2. Update the tests to cover your changes.
3. The PR should pass all automated checks and tests.
4. Your PR needs to be reviewed and approved by the maintainers.

## Project Structure

Please follow the project structure when contributing:

```
dynamiccompactdetect/
├── data/               # Dataset files and test images
├── docs/               # Documentation files
├── models/             # Model files
├── results/            # Benchmark results and comparisons
├── scripts/            # Utility scripts
├── src/                # Source code
├── tests/              # Test scripts
└── visualizations/     # Visualization outputs
```

## Development Environment

1. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install development dependencies:
   ```
   pip install -e ".[dev]"
   ```

## Reporting Bugs

If you find a bug, please create an issue on the GitHub repository with:

1. A clear title and description
2. Steps to reproduce the issue
3. Expected behavior
4. Actual behavior
5. Any relevant logs or screenshots

## Feature Requests

Feature requests are welcome. Please create an issue with:

1. A clear title and description
2. Explanation of why this feature would be useful
3. Any ideas for implementation

## Questions?

If you have any questions, feel free to create an issue with the "question" label.

Thank you for your contributions!

—Abhilash Chadhar and Divya Athya 