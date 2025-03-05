# Python Tender Normalizer

This repository contains the Python-based normalizer service for processing tenders from various sources.

## Latest Updates

The latest version (commit `b29d864`) includes important fixes for:

1. Country validation for all sources (SAM.gov, TED EU, UNGM, AIIB, etc.)
2. Title extraction for ADB and IADB tenders
3. Date parsing for AFD tenders and handling "Unknown" dates for AFDB
4. Performance stats logging method

## Deployment History

- **Version 1.1.0** (Latest) - Updated on June 25, 2023 with validation fixes and improved error handling

## Deployment Instructions

To deploy the latest version:

1. **Pull the repository**:
   ```bash
   git clone https://github.com/Berniel12/pynormalizer.git
   cd pynormalizer
   git checkout main
   ```

2. **Build the Docker image**:
   ```bash
   docker build -t pynormalizer:latest .
   ```

3. **Tag and push to your registry** (replace with your actual registry URL):
   ```bash
   docker tag pynormalizer:latest your-registry.com/pynormalizer:latest
   docker push your-registry.com/pynormalizer:latest
   ```

4. **Update your deployment** to use the latest image.

For more detailed deployment instructions, see the [DEPLOYMENT.md](DEPLOYMENT.md) file.

## Environment Variables

The normalizer requires the following environment variables:

- `SUPABASE_URL`: URL of your Supabase instance
- `SUPABASE_KEY`: API key for Supabase
- `OPENAI_API_KEY`: OpenAI API key (if using OpenAI capabilities)
- `TEST_MODE`: Set to 'True' to run in test mode with a limited number of tenders

## Troubleshooting

If you encounter validation errors, ensure you're using the latest version of the code, which includes robust fallback mechanisms for all required fields.

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