# Python Normalizer

A Python service for normalizing tender data using various methods, including LLM-based normalization.

## Features

- Normalizes tender data from various sources into a consistent format
- Uses multiple normalization methods with fallback mechanisms:
  1. PydanticAI-based normalization (primary method)
  2. DirectNormalizer using OpenAI API (fallback for PydanticAI serialization issues)
  3. MockNormalizer for testing and fallback when API calls fail
  4. Direct parsing as a final fallback
- Handles special characters and complex nested data structures
- Provides detailed logging and debugging information

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```python
from src.models.tender import RawTender
from src.services.normalizer import TenderNormalizer

# Create a tender normalizer
normalizer = TenderNormalizer()

# Create a raw tender
tender = RawTender(
    id="tender-001",
    source_table="sam_gov",
    title="Example Tender",
    description="This is an example tender description.",
    country="United States",
    organization_name="Example Organization"
)

# Normalize the tender
result = normalizer.normalize_tender_sync(tender)

# Access the normalized data
normalized_data = result["normalized_data"]
method_used = result["method"]
processing_time = result["processing_time"]
```

## Fallback Mechanism

The normalizer implements a robust fallback mechanism to ensure that tender data is always normalized, even when the primary methods fail:

1. **PydanticAI Normalization**: The primary method uses PydanticAI to normalize the tender data. This provides strong type validation and structured output.

2. **DirectNormalizer Fallback**: If PydanticAI fails due to serialization issues (common with the "Expected code to be unreachable" error in PydanticAI 0.0.31), the normalizer falls back to DirectNormalizer, which makes direct API calls to OpenAI.

3. **MockNormalizer Fallback**: If both PydanticAI and DirectNormalizer fail (e.g., due to API authentication errors), the normalizer falls back to MockNormalizer, which provides a mock implementation for testing and development.

4. **Direct Parsing Fallback**: As a final fallback, the normalizer uses direct parsing to extract basic fields from the raw tender data.

This multi-level fallback mechanism ensures that the normalization process is robust and can handle various error conditions.

## Testing

The repository includes several test scripts to verify the functionality of the normalizer:

- `scripts/test_llm.py`: Tests the PydanticAI-based normalization
- `scripts/test_direct.py`: Tests the DirectNormalizer implementation
- `scripts/test_mock.py`: Tests the MockNormalizer implementation
- `scripts/test_fallback.py`: Tests the fallback mechanism
- `scripts/test_all_normalizers.py`: Tests all normalization methods

To run the tests:

```bash
python -m scripts.test_all_normalizers
```

## Configuration

The normalizer can be configured using environment variables or by modifying the `src/config.py` file:

- `OPENAI_API_KEY`: OpenAI API key for LLM-based normalization
- `OPENAI_MODEL`: OpenAI model to use (default: "gpt-4o-mini")

> **Important Note**: This project uses `gpt-4o-mini` as the default model for all API calls to minimize costs. Other models like `gpt-4o`, `gpt-4-turbo`, or `gpt-4` are significantly more expensive and should **not** be used for this task unless there's a specific requirement. The `gpt-4o-mini` model provides sufficient accuracy for tender normalization at a fraction of the cost.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 