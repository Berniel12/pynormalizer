#!/usr/bin/env python3
"""
Script to test the tender normalizer locally before Apify deployment.
This is useful to verify everything works as expected before uploading to Apify.
"""
import asyncio
import json
import os
import sys
from pathlib import Path

# Add the parent directory to the path so we can import the src package
sys.path.append(str(Path(__file__).parent.parent))

from dotenv import load_dotenv

# Load environment variables from .env for local testing
load_dotenv()

from src.config import settings
from src.models.tender import RawTender
from src.services.normalizer import normalizer


async def test_normalizer():
    """Test the normalizer with a sample tender."""
    
    # Check if we have the required environment variables
    if not settings.openai_api_key.get_secret_value():
        print("ERROR: OPENAI_API_KEY environment variable is not set")
        print("Create a .env file in the python_normalizer directory with:")
        print("OPENAI_API_KEY=your_openai_api_key")
        return

    # Create a sample tender for testing
    sample_tender = RawTender(
        id="test-123",
        source_table="test_source",
        title="Sample Construction Project",
        description="This is a test tender for a construction project in Ghana. The project involves building a new school.",
        country="Ghana",
        publication_date="2023-01-01",
        deadline_date="2023-05-01",
        organization_name="Ministry of Education",
        value="1000000",
        currency="USD",
    )
    
    print(f"\nTesting normalizer with sample tender:")
    print(f"ID: {sample_tender.id}")
    print(f"Title: {sample_tender.title}")
    print(f"Description: {sample_tender.description[:50]}...")
    
    # Normalize the tender
    should_use_llm, reason = normalizer._should_use_llm(sample_tender)
    print(f"\nShould use LLM: {should_use_llm} ({reason})")
    
    print("\nNormalizing tender with LLM...")
    result = await normalizer.normalize_tender(sample_tender)
    
    print(f"\nNormalization result:")
    print(f"Success: {result.success}")
    print(f"Method used: {result.method_used}")
    print(f"Processing time: {result.processing_time:.2f} seconds")
    print(f"Fields before: {result.fields_before}")
    print(f"Fields after: {result.fields_after}")
    print(f"Improvement percentage: {result.improvement_percentage:.2f}%")
    
    if result.success and result.normalized_tender:
        print("\nNormalized tender:")
        print(f"Title: {result.normalized_tender.title}")
        print(f"Country: {result.normalized_tender.country}")
        print(f"Status: {result.normalized_tender.status}")
        print(f"Tender type: {result.normalized_tender.tender_type}")
        
        # Print out all fields in the normalized tender
        print("\nAll normalized fields:")
        for key, value in result.normalized_tender.model_dump().items():
            if isinstance(value, (dict, list)):
                value = json.dumps(value)[:50] + "..." if len(json.dumps(value)) > 50 else json.dumps(value)
            print(f"  {key}: {value}")
    else:
        print(f"\nNormalization failed: {result.error}")


if __name__ == "__main__":
    asyncio.run(test_normalizer()) 