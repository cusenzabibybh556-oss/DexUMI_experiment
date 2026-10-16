# Contributing to LinkerHand ROS Teleop SDK

Thank you for your interest in contributing to this project!

## Getting Started

### Prerequisites

- Python 3.8+ (ROS1 Noetic) or Python 3.10+ (ROS2 Humble/Jazzy)
- ROS Noetic (ROS1) or ROS2 Humble/Jazzy
- Git

### Development Setup

1. Clone the repository:
   ```bash
   git clone ssh://git@git.linkerhub.work:12222/linker-bot/linkerhand_telop_python.git
   cd linkerhand_telop_python
   ```
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   # or
   .\venv\Scripts\activate  # Windows
   ```
3. Install development dependencies:
   ```bash
   pip install -r new/ros2/src/requirements.txt
   pip install ruff pytest
   ```

## Development Workflow

### Branch Naming

Use descriptive branch names with prefixes:

- `feature/` - New features (e.g., `feature/add-new-glove-support`)
- `fix/` - Bug fixes (e.g., `fix/retarget-calculation-error`)
- `docs/` - Documentation changes (e.g., `docs/update-installation-guide`)
- `refactor/` - Code refactoring (e.g., `refactor/simplify-config-loading`)

### Commit Messages

Follow conventional commit format:

```
type(scope): brief description

[optional body]
[optional footer]
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

Examples:
- `feat(linkerhand): add support for LinkerHand L30`
- `fix(retarget): correct joint angle calculation`
- `docs(readme): update installation steps`

### Pull Request Process

1. Create a feature branch from `main`
2. Make your changes
3. Run tests and linting:
   ```bash
   ruff check new/
   pytest tests/  # if tests exist
   ```
4. Commit your changes with clear messages
5. Push to the repository and create a PR
6. Ensure CI passes
7. Wait for code review

### Code Style

- Follow PEP 8 guidelines
- Use type hints where appropriate
- Write docstrings for public functions and classes
- Keep functions focused and reasonably sized

### Testing

- Add tests for new functionality
- Ensure existing tests pass
- Test with actual hardware when possible

## Directory Structure

```
new/
├── ros1/src/           # ROS1 package
├── ros2/src/           # ROS2 package
├── LICENSE
├── README.md
├── CONTRIBUTING.md
├── SECURITY.md
└── CODE_OF_CONDUCT.md
```

## Questions?

Open an issue for:
- Bug reports
- Feature requests
- Questions about usage

Thank you for contributing!
