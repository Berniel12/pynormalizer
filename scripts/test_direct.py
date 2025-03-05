#!/usr/bin/env python3
import argparse
import asyncio
import json
import logging
import os
import sys
import time
import traceback
from pathlib import Path

import openai

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.config import settings
from src.services.direct_normalizer import DirectNormalizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("direct_tester")

def create_test_tender():
    """Create a test tender with various special characters and nested data."""
    logger.info("Creating test tender")
    
    return {
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

async def test_direct_normalization(api_key):
    """Test the DirectNormalizer implementation."""
    try:
        # Create a test tender
        tender = create_test_tender()
        logger.info(f"Created test tender: {tender['id']} from {tender['source_table']}")
        
        # Save input data for debugging
        debug_dir = Path("debug_dumps")
        debug_dir.mkdir(exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d%H%M%S")
        input_filename = f"input_{tender['source_table']}_{tender['id']}_{timestamp}.json"
        with open(debug_dir / input_filename, "w") as f:
            json.dump(tender, f, indent=2)
        logger.info(f"Saved input data to {debug_dir / input_filename}")
        
        # Initialize the DirectNormalizer
        logger.info(f"Initializing DirectNormalizer with API key: {api_key[:5]}...{api_key[-4:] if len(api_key) > 8 else ''}")
        normalizer = DirectNormalizer(
            api_key=api_key,
            model=settings.openai_model if hasattr(settings, 'openai_model') else "gpt-4o-mini"
        )
        
        # Normalize the tender
        logger.info("Starting normalization process...")
        start_time = time.time()
        
        try:
            result = await normalizer.normalize_tender(tender, save_debug=True)
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            if result and "normalized_data" in result and result["normalized_data"]:
                logger.info(f"✅ SUCCESS: Tender normalized in {processing_time:.2f} seconds")
                logger.info(f"Method used: {result.get('method', 'unknown')}")
                
                # Check for missing fields
                missing_fields = result.get("missing_fields", [])
                if missing_fields:
                    logger.warning(f"Missing fields: {', '.join(missing_fields)}")
                
                # Check for notes
                notes = result.get("notes")
                if notes:
                    logger.info(f"Notes: {notes}")
                
                # Save output data for debugging
                output_filename = f"output_{tender['source_table']}_{tender['id']}_{timestamp}.json"
                with open(debug_dir / output_filename, "w") as f:
                    json.dump(result, f, indent=2)
                logger.info(f"Saved output data to {debug_dir / output_filename}")
                
                # Print the normalized data
                logger.info("Normalized data:")
                logger.info(json.dumps(result["normalized_data"], indent=2))
                
                return True
            else:
                logger.error("❌ FAILURE: Normalization returned empty or invalid result")
                return False
                
        except openai.AuthenticationError as e:
            logger.error(f"❌ FAILURE: OpenAI API authentication error: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"❌ FAILURE: Error during normalization: {str(e)}")
            
            # Save error data for debugging
            error_data = {
                "error": str(e),
                "traceback": traceback.format_exc()
            }
            error_filename = f"error_{tender['source_table']}_{tender['id']}_{timestamp}.json"
            with open(debug_dir / error_filename, "w") as f:
                json.dump(error_data, f, indent=2)
            logger.info(f"Saved error data to {debug_dir / error_filename}")
            
            return False
            
    except Exception as e:
        logger.error(f"❌ FAILURE: Unexpected error: {str(e)}")
        return False
    finally:
        # Check for error logs in debug_dumps directory
        debug_dir = Path("debug_dumps")
        error_files = list(debug_dir.glob(f"error_*_{tender['id']}_*.json"))
        if error_files:
            for error_file in error_files:
                try:
                    with open(error_file, "r") as f:
                        error_data = json.load(f)
                    logger.info(f"Contents of {error_file.name}:")
                    logger.info(json.dumps(error_data, indent=2))
                except Exception as e:
                    logger.error(f"Error reading error file {error_file}: {str(e)}")
        
        # Log performance stats
        logger.info("DirectNormalizer test completed")

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Test the DirectNormalizer implementation")
    parser.add_argument("--api-key", help="OpenAI API key")
    args = parser.parse_args()
    
    # Get the API key from command line arguments or environment variables
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY") or settings.openai_api_key.get_secret_value()
    
    if not api_key:
        logger.error("No OpenAI API key provided. Please set the OPENAI_API_KEY environment variable or use the --api-key argument.")
        sys.exit(1)
    
    logger.info(f"Using OpenAI API key: {api_key[:5]}...{api_key[-4:] if len(api_key) > 8 else ''}")
    
    # Create debug directory if it doesn't exist
    os.makedirs("debug_dumps", exist_ok=True)
    
    # Run the test
    asyncio.run(test_direct_normalization(api_key)) 