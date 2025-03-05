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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("fallback_tester")

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
    
    # Add null bytes to force serialization errors
    tender_data["description"] += "\u0000" * 100
    
    return RawTender(**tender_data)

async def test_fallback_to_mock():
    """Test the fallback mechanism to MockNormalizer."""
    logger.info("=== Testing Fallback to MockNormalizer ===")
    
    # Create debug directory
    debug_dir = Path("debug_dumps")
    debug_dir.mkdir(exist_ok=True)
    
    # Create test tender with problematic data
    tender = create_test_tender()
    logger.info(f"Created test tender: {tender.id} from {tender.source_table}")
    
    # Save input data for debugging
    timestamp = time.strftime("%Y%m%d%H%M%S")
    input_filename = f"input_fallback_{tender.source_table}_{tender.id}_{timestamp}.json"
    with open(debug_dir / input_filename, "w") as f:
        json.dump(tender.model_dump(), f, indent=2, default=str)
    logger.info(f"Saved input data to {debug_dir / input_filename}")
    
    # Initialize the TenderNormalizer
    normalizer = TenderNormalizer()
    logger.info("Initialized TenderNormalizer")
    
    # Set an invalid API key to force DirectNormalizer to fail
    os.environ["OPENAI_API_KEY"] = "sk-invalid-key"
    
    # Normalize the tender
    logger.info("Starting normalization process...")
    start_time = time.time()
    
    try:
        result = await normalizer.normalize_tender(tender, save_debug=True)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Log results
        logger.info(f"Method used: {result.get('method', 'unknown')}")
        logger.info(f"Processing time: {processing_time:.2f} seconds")
        
        if result.get('error'):
            logger.error(f"Error: {result.get('error')}")
            return False
        else:
            logger.info("Normalization successful")
            logger.info(f"Normalized data: {json.dumps(result.get('normalized_data', {}), indent=2)}")
            
            # Check if MockNormalizer was used
            if result.get('method') == 'mock_normalizer':
                logger.info("✅ SUCCESS: Fallback to MockNormalizer worked correctly")
                return True
            else:
                logger.warning(f"⚠️ WARNING: Expected method 'mock_normalizer', but got '{result.get('method')}'")
                return False
    except Exception as e:
        logger.error(f"❌ ERROR: Unexpected exception: {str(e)}")
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    # Run the test
    asyncio.run(test_fallback_to_mock()) 