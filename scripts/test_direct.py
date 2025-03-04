#!/usr/bin/env python3
"""
Script to test the DirectNormalizer implementation.
"""
import asyncio
import json
import os
import sys
import logging
from datetime import datetime
import time
import random
import string
import openai
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("direct_normalizer_test")

# Add the parent directory to the path so we can import the src package
sys.path.append(".")

# Check if OpenAI API key is provided
if len(sys.argv) > 1:
    api_key = sys.argv[1]
    logger.info("Using API key from command line argument")
else:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.error("No OpenAI API key provided. Please provide it as an argument or set OPENAI_API_KEY environment variable.")
        sys.exit(1)
    logger.info("Using API key from environment variable")

# Import the DirectNormalizer
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.services.direct_normalizer import DirectNormalizer

def create_test_tender(tender_id="test-sam_gov-001", source="sam_gov"):
    """Create a test tender with potentially problematic data."""
    return {
        "id": tender_id,
        "source_table": source,
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
        "publication_date": datetime.now().isoformat(),
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

async def test_direct_normalization():
    """Test the DirectNormalizer implementation."""
    logger.info("Starting DirectNormalizer test")
    
    # Create test tender
    test_tender = create_test_tender()
    logger.info(f"Created test tender: {test_tender['id']} from {test_tender['source_table']}")
    
    # Initialize the DirectNormalizer
    normalizer = DirectNormalizer(
        api_key=api_key,
        model="gpt-4o",
        max_retries=2,
        retry_delay=1
    )
    
    # Normalize tender
    logger.info("Starting normalization process...")
    try:
        normalized_data, method, processing_time = await normalizer.normalize_tender(test_tender)
        
        # Check if normalization was successful
        if method == "llm":
            logger.info(f"✅ SUCCESS: Normalization completed in {processing_time:.2f}s")
            logger.info(f"Normalized tender data: {json.dumps(normalized_data['tender'], indent=2)}")
            
            # Save normalized data to file
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            output_file = f"debug_dumps/normalized_{test_tender['source_table']}_{test_tender['id']}_{timestamp}.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(normalized_data, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved normalized data to {output_file}")
            
            # Check for missing fields
            if "missing_fields" in normalized_data and normalized_data["missing_fields"]:
                logger.warning(f"Missing fields: {', '.join(normalized_data['missing_fields'])}")
            
            # Check for notes
            if "notes" in normalized_data and normalized_data["notes"]:
                logger.info(f"Notes: {normalized_data['notes']}")
        else:
            logger.error(f"❌ FAILURE: Normalization failed")
            logger.error(f"Error message: {normalized_data.get('notes', 'Unknown error')}")
            logger.error(f"Method used: {method}")
            logger.error(f"Processing time: {processing_time:.2f}s")
    except openai.AuthenticationError as e:
        logger.error(f"❌ AUTHENTICATION ERROR: {str(e)}")
        logger.info("Please check your OpenAI API key and try again. The current key is invalid.")
    except Exception as e:
        logger.error(f"❌ EXCEPTION: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
    
    # Check for any error logs in debug_dumps directory
    for filename in os.listdir("debug_dumps"):
        if filename.startswith("error_") and test_tender["source_table"] in filename and test_tender["id"] in filename:
            logger.info(f"Found error log: {filename}")
            try:
                with open(os.path.join("debug_dumps", filename), "r", encoding="utf-8") as f:
                    error_data = json.load(f)
                    logger.error(f"Error details: {error_data.get('error', 'Unknown error')}")
                    logger.error(f"Traceback: {error_data.get('traceback', 'No traceback available')}")
            except Exception as e:
                logger.error(f"Failed to read error log: {str(e)}")
    
    # Log performance statistics
    normalizer._log_performance_stats()
    
    logger.info("DirectNormalizer test completed")

if __name__ == "__main__":
    # Create debug directory if it doesn't exist
    os.makedirs("debug_dumps", exist_ok=True)
    
    # Run the test
    asyncio.run(test_direct_normalization()) 