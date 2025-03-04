# Tender Normalizer (Python Version)

A Python-based implementation of the tender normalization system using Pydantic and PydanticAI.

## Overview

This project uses Pydantic and PydanticAI to normalize tender data from various sources. The goal is to provide a more robust, type-safe implementation with enhanced LLM integration compared to the Node.js version.

## Features

- Strong data validation using Pydantic models
- Structured LLM interactions using PydanticAI
- Improved handling of critical fields (title, country, description)
- Robust fallback mechanisms for LLM timeouts
- Monitoring and instrumentation with Logfire

## Project Structure

```
python_normalizer/
├── src/                 # Source code
│   ├── models/          # Pydantic data models
│   ├── services/        # Business logic services
│   ├── config.py        # Configuration settings
│   └── main.py          # Application entry point
├── tests/               # Test cases
├── scripts/             # Utility scripts
└── pyproject.toml       # Project dependencies and configuration
```

## Setup

1. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```
   pip install -e ".[dev]"
   ```

3. Create a `.env` file with required environment variables:
   ```
   SUPABASE_URL=your_supabase_url
   SUPABASE_KEY=your_supabase_key
   OPENAI_API_KEY=your_openai_key
   LOGFIRE_TOKEN=your_logfire_token
   ```

## Usage

To process tenders:

```python
from src.main import process_all_tenders

# Process all unprocessed tenders
process_all_tenders()
```

## Development

This project uses:
- Black and isort for code formatting
- Mypy for type checking
- Pytest for testing
- Ruff for linting

Run the test suite:
```
pytest
``` 