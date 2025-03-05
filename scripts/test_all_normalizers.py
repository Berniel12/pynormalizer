#!/usr/bin/env python3
import asyncio
import json
import logging
import os
import sys
import time
import traceback
from pathlib import Path

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.config import settings
from src.models.tender import RawTender
from src.services.normalizer import TenderNormalizer
from src.services.direct_normalizer import DirectNormalizer
from src.services.mock_normalizer import MockNormalizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("normalizer_tester")

def create_test_tender():
    """Create a test tender with various special characters and nested data."""
    logger.info("Creating test tender")
    
    tender_data = {
        "id": "test-sam_gov-001",
        "source_table": "sam_gov",
        "title": "Test tender with special chars: apostrophe's, quotes\", and em-dash—plus accented chars éèçà",
        "description": """
        This is a test tender description with various special characters:
        • Bullets and lists
        • Single quotes: ' and ' and ‛
        • Double quotes: " and " and „
        • Dashes: - and – and — 
        • Other symbols: … © ® ™ € £ ¥ ÷ × 
        • Accented: àáâãäåçèéêëìíîïñòóôõöùúûüýÿ
        """,
        "publication_date": time.strftime("%Y-%m-%dT%H:%M:%S.%f"),
        "deadline_date": None,
        "country": "United States",
        "country_code": None,
        "location": None,
        "organization_name": "Test Organization & Partners, LLC.",
        "organization_id": None,
        "title_english": None,
        "description_english": None,
        "organization_name_english": None,
        "status": None,
        "tender_type": None,
        "value": None,
        "currency": None,
        "language": None,
        "url": "https://example.com/tender/123",
        "link": None,
        "web_link": None,
        "source_id": None,
        "source_data": None,
        "created_at": None,
        "updated_at": None,
        "normalized": False,
        "processed": False,
        "nested_data": {
            "complex_field": {
                "array_with_mixed_types": [
                    1,
                    "two",
                    3.0,
                    None,
                    {
                        "key": "value"
                    }
                ],
                "empty_list": [],
                "problematic_string": "Problem string with null bytes: \u0000\u0000\u0000"
            }
        }
    }
    
    return RawTender(**tender_data)

async def test_pydantic_ai_normalizer(tender: RawTender):
    """Test the PydanticAI normalizer."""
    logger.info("=== Testing PydanticAI Normalizer ===")
    
    try:
        # Initialize the TenderNormalizer
        normalizer = TenderNormalizer()
        logger.info("Initialized TenderNormalizer")
        
        # Normalize the tender
        start_time = time.time()
        result = await normalizer.normalize_tender(tender, save_debug=True)
        end_time = time.time()
        
        # Log results
        logger.info(f"Method used: {result.get('method', 'unknown')}")
        logger.info(f"Processing time: {result.get('processing_time', end_time - start_time):.2f} seconds")
        
        if result.get('error'):
            logger.error(f"Error: {result.get('error')}")
            return False
        else:
            logger.info("Normalization successful")
            return True
    except Exception as e:
        logger.error(f"Error in PydanticAI normalization: {str(e)}")
        logger.error(traceback.format_exc())
        return False

async def test_direct_normalizer(tender: RawTender, api_key: str):
    """Test the DirectNormalizer."""
    logger.info("=== Testing DirectNormalizer ===")
    
    try:
        # Initialize the DirectNormalizer
        normalizer = DirectNormalizer(
            api_key=api_key,
            model=settings.openai_model if hasattr(settings, 'openai_model') else "gpt-4o-mini"
        )
        logger.info("Initialized DirectNormalizer")
        
        # Normalize the tender
        start_time = time.time()
        result_tuple = await normalizer.normalize_tender(tender.model_dump(), save_debug=True)
        end_time = time.time()
        
        # DirectNormalizer returns a tuple of (normalized_data, method, processing_time)
        normalized_data, method, processing_time = result_tuple
        
        # Log results
        logger.info(f"Method used: {method}")
        logger.info(f"Processing time: {processing_time:.2f} seconds")
        
        if not normalized_data or not normalized_data.get('tender', {}):
            logger.error("Error: No valid result returned")
            return False
        else:
            logger.info("Normalization successful")
            logger.info(f"Normalized data: {json.dumps(normalized_data.get('tender', {}), indent=2)}")
            return True
    except Exception as e:
        logger.error(f"Error in DirectNormalizer: {str(e)}")
        logger.error(traceback.format_exc())
        return False

async def test_mock_normalizer(tender: RawTender):
    """Test the MockNormalizer."""
    logger.info("=== Testing MockNormalizer ===")
    
    try:
        # Initialize the MockNormalizer
        normalizer = MockNormalizer()
        logger.info("Initialized MockNormalizer")
        
        # Normalize the tender
        start_time = time.time()
        result = await normalizer.normalize_tender(tender.model_dump())
        end_time = time.time()
        
        # Log results
        logger.info(f"Method used: {result.get('method', 'mock')}")
        logger.info(f"Processing time: {result.get('processing_time', end_time - start_time):.2f} seconds")
        
        if not result or not result.get('normalized_data'):
            logger.error("Error: No valid result returned")
            return False
        else:
            logger.info("Normalization successful")
            logger.info(f"Normalized data: {json.dumps(result.get('normalized_data'), indent=2)}")
            return True
    except Exception as e:
        logger.error(f"Error in MockNormalizer: {str(e)}")
        logger.error(traceback.format_exc())
        return False

async def test_fallback_mechanism(tender: RawTender, api_key: str):
    """Test the fallback mechanism in the TenderNormalizer."""
    logger.info("=== Testing Fallback Mechanism ===")
    
    try:
        # Initialize the TenderNormalizer
        normalizer = TenderNormalizer()
        logger.info("Initialized TenderNormalizer")
        
        # Force PydanticAI to fail by modifying the tender
        # This is a hack to trigger the fallback mechanism
        tender_dict = tender.model_dump()
        tender_dict["description"] = tender_dict["description"] + "\u0000" * 100  # Add null bytes to force serialization error
        modified_tender = RawTender(**tender_dict)
        
        # Normalize the tender
        start_time = time.time()
        result = await normalizer.normalize_tender(modified_tender, save_debug=True)
        end_time = time.time()
        
        # Log results
        logger.info(f"Method used: {result.get('method', 'unknown')}")
        logger.info(f"Processing time: {result.get('processing_time', end_time - start_time):.2f} seconds")
        
        if result.get('error'):
            logger.error(f"Error: {result.get('error')}")
            return False
        else:
            logger.info("Normalization successful with fallback")
            logger.info(f"Normalized data: {json.dumps(result.get('normalized_data'), indent=2)}")
            return True
    except Exception as e:
        logger.error(f"Error in fallback mechanism: {str(e)}")
        logger.error(traceback.format_exc())
        return False

async def main():
    """Run all tests."""
    # Create debug directory
    debug_dir = Path("debug_dumps")
    debug_dir.mkdir(exist_ok=True)
    
    # Get API key
    api_key = os.environ.get("OPENAI_API_KEY") or settings.openai_api_key.get_secret_value()
    if not api_key:
        logger.error("No OpenAI API key provided. Using placeholder.")
        api_key = "sk-placeholder"
    
    # Create test tender
    tender = create_test_tender()
    logger.info(f"Created test tender: {tender.id} from {tender.source_table}")
    
    # Test each normalizer
    pydantic_result = await test_pydantic_ai_normalizer(tender)
    direct_result = await test_direct_normalizer(tender, api_key)
    mock_result = await test_mock_normalizer(tender)
    fallback_result = await test_fallback_mechanism(tender, api_key)
    
    # Log summary
    logger.info("=== Test Summary ===")
    logger.info(f"PydanticAI Normalizer: {'SUCCESS' if pydantic_result else 'FAILURE'}")
    logger.info(f"DirectNormalizer: {'SUCCESS' if direct_result else 'FAILURE'}")
    logger.info(f"MockNormalizer: {'SUCCESS' if mock_result else 'FAILURE'}")
    logger.info(f"Fallback Mechanism: {'SUCCESS' if fallback_result else 'FAILURE'}")
    
    # Check for error logs in debug_dumps directory
    error_files = list(debug_dir.glob(f"error_*_{tender.id}_*.json"))
    if error_files:
        logger.info("=== Error Files ===")
        for error_file in error_files:
            logger.info(f"Found error file: {error_file.name}")

if __name__ == "__main__":
    # Run the tests
    asyncio.run(main()) 