#!/usr/bin/env python3
import asyncio
import json
import logging
import os
import sys
import time
import traceback
from datetime import datetime
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

def create_test_tender_with_special_chars():
    """Create a test tender with special characters to test parsing mechanism."""
    tender_data = {
        "id": "test-special-chars-001",
        "source_table": "test_source",
        "title": "Test tender with special characters: ñ, é, ü, ç, ß, 你好, こんにちは",
        "description": """This is a test tender with various special characters and formatting:
        • Bullet point 1
        • Bullet point 2
        
        **Bold text** and *italic text*
        
        Table:
        | Column 1 | Column 2 |
        |----------|----------|
        | Value 1  | Value 2  |
        
        Code: `print("Hello World")`
        
        > Blockquote text
        
        ---
        
        # Heading 1
        ## Heading 2
        
        [Link text](https://example.com)
        """,
        "publication_date": datetime.now().date().isoformat(),
        "deadline_date": datetime.now().date().isoformat(),
        "country": "Test Country",
        "organization_name": "Test Organization",
        "url": "https://example.com/test/tender/001",
        "source_data": {
            "complex_field": {
                "nested_field_1": "value 1",
                "nested_field_2": "value 2",
                "nested_array": [1, 2, 3, 4, 5]
            },
            "array_field": ["item 1", "item 2", "item 3"],
            "boolean_field": True,
            "number_field": 12345.67
        }
    }
    
    logger.info(f"Created test tender with special characters: {tender_data['id']}")
    return RawTender(**tender_data)

async def test_direct_parsing():
    """Test the direct parsing approach for tender normalization."""
    logger.info("=== Testing direct parsing ===")
    
    # Create a test tender with special characters
    tender = create_test_tender_with_special_chars()
    
    # Initialize the TenderNormalizer
    normalizer = TenderNormalizer()
    logger.info("Initialized TenderNormalizer")
    
    # Normalize the tender
    logger.info(f"Attempting to normalize tender: {tender.id}")
    start_time = time.time()
    
    try:
        result = await normalizer.normalize_tender(tender)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Check if direct parsing was used
        if result.get('method') == "direct_parsing":
            logger.info(f"✅ SUCCESS: Direct parsing worked correctly")
            logger.info(f"Processing time: {processing_time:.2f} seconds")
            logger.info(f"Normalized data: {json.dumps(result.get('normalized_data', {}), indent=2)}")
        else:
            logger.error(f"❌ ERROR: Expected method to be direct_parsing, but got: {result.get('method')}")
    except Exception as e:
        logger.error(f"❌ ERROR: Normalization failed: {str(e)}")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    asyncio.run(test_direct_parsing()) 