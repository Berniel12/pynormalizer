import asyncio
import json
import logging
import os
import sys
import time
import traceback
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.services.mock_normalizer import MockNormalizer
from src.config import NormalizerConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("mock_tester")

def create_test_tender():
    """Create a test tender with various special characters and nested data."""
    tender = {
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
        "publication_date": time.strftime("%Y-%m-%dT%H:%M:%S.f"),
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
                "array_with_mixed_types": [1, "two", 3.0, None, {"key": "value"}],
                "empty_list": [],
                "problematic_string": "Problem string with null bytes: \0\0\0"
            }
        }
    }
    return tender

async def test_mock_normalization():
    """Test the mock normalization process."""
    # Create a test tender
    tender = create_test_tender()
    tender_id = tender["id"]
    source = tender["source_table"]
    logger.info(f"Created test tender: {tender_id} from {source}")
    
    # Create debug directory if it doesn't exist
    debug_dir = Path("debug_dumps")
    debug_dir.mkdir(exist_ok=True)
    
    # Save input data for debugging
    timestamp = time.strftime("%Y%m%d%H%M%S")
    input_file = debug_dir / f"input_{source}_{tender_id}_{timestamp}.json"
    with open(input_file, "w") as f:
        json.dump(tender, f, indent=2)
    logger.info(f"Saved input data to {input_file}")
    
    # Initialize the MockNormalizer
    config = NormalizerConfig(save_debug_data=True)
    normalizer = MockNormalizer(config=config)
    logger.info("Initialized MockNormalizer")
    
    # Start the normalization process
    logger.info("Starting mock normalization process...")
    start_time = time.time()
    
    try:
        # Normalize the tender
        result = await normalizer.normalize_tender(tender)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Log the result
        if result and result.get("normalized_data"):
            logger.info(f"✅ SUCCESS: Normalized tender in {processing_time:.2f} seconds")
            logger.info(f"Method: {result.get('method', 'unknown')}")
            logger.info(f"Missing fields: {result.get('missing_fields', [])}")
            logger.info(f"Notes: {result.get('notes', '')}")
            
            # Save normalized data for debugging
            output_file = debug_dir / f"output_{source}_{tender_id}_{timestamp}.json"
            with open(output_file, "w") as f:
                json.dump(result, f, indent=2)
            logger.info(f"Saved normalized data to {output_file}")
            
            # Print the normalized data
            logger.info("Normalized data:")
            logger.info(json.dumps(result["normalized_data"], indent=2))
        else:
            logger.error("❌ FAILURE: Normalization returned empty or invalid result")
    
    except Exception as e:
        logger.error(f"❌ ERROR: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Save error data for debugging
        error_file = debug_dir / f"error_{source}_{tender_id}_{timestamp}.json"
        with open(error_file, "w") as f:
            json.dump({
                "error": str(e),
                "traceback": traceback.format_exc()
            }, f, indent=2)
        logger.info(f"Saved error data to {error_file}")
    
    logger.info("Mock normalization test completed")

if __name__ == "__main__":
    # Run the test
    asyncio.run(test_mock_normalization()) 