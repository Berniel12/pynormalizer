import asyncio
import os
import json
import logging
from datetime import datetime
from typing import Dict, Any
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("llm_tester")

# Set required environment variables for the normalizer
os.environ["OPENAI_API_KEY"] = "sk-" 
# Set your API key here if needed or add it via command line when running the script

import sys
sys.path.append(".")  # Add current directory to path

from src.models.tender import RawTender, NormalizationResult
from src.services.normalizer import TenderNormalizer, NormalizationInput, NormalizationOutput

def create_test_tender(source: str = "sam_gov") -> RawTender:
    """
    Create a test tender with potentially problematic data.
    
    Args:
        source: The source table to simulate
        
    Returns:
        A raw tender object
    """
    current_time = datetime.now().isoformat()
    
    # Create test data with various potential issues
    tender_data: Dict[str, Any] = {
        "id": f"test-{source}-001",
        "source_table": source,
        "title": f"Test tender with special chars: apostrophe's, quotes\", and em-dash—plus accented chars éèçà",
        "description": """
        This is a test tender description with various special characters:
        • Bullets and lists
        • Single quotes: ' and ' and ‛
        • Double quotes: " and " and „
        • Dashes: - and – and — 
        • Other symbols: … © ® ™ € £ ¥ ÷ × 
        • Accented: àáâãäåçèéêëìíîïñòóôõöùúûüýÿ
        """,
        "publication_date": current_time,
        "country": "United States",
        "organization_name": "Test Organization & Partners, LLC.",
        "url": "https://example.com/tender/123",
        # Add other fields that might be problematic
        "nested_data": {
            "complex_field": {
                "array_with_mixed_types": [1, "two", 3.0, None, {"key": "value"}],
                "empty_list": [],
                "problematic_string": "Problem string with null bytes: \0\0\0",
            }
        }
    }
    
    # Return as RawTender object
    return RawTender(**tender_data)

async def normalize_tender(normalizer: TenderNormalizer, tender: RawTender) -> NormalizationResult:
    """
    Normalize a tender using the TenderNormalizer.
    This is a wrapper around the _normalize_with_llm method.
    
    Args:
        normalizer: The TenderNormalizer instance
        tender: The raw tender to normalize
        
    Returns:
        A NormalizationResult object
    """
    logger.info(f"Normalizing tender {tender.id} from {tender.source_table}")
    
    # Create input for the normalizer
    tender_dict = tender.model_dump()
    input_data = NormalizationInput(
        id=tender.id,
        source_table=tender.source_table,
        title=tender.title,
        description=tender.description,
        country=tender.country,
        organization_name=tender.organization_name,
        raw_tender=tender_dict
    )
    
    # Start timing
    start_time = datetime.now()
    
    try:
        # Create a custom wrapper for _normalize_with_llm
        async def custom_normalize_with_llm():
            try:
                # Debug the input data for LLM
                logger.debug(f"Input data for LLM normalization: {tender.id}")
                # Log essential fields with their types and lengths if string
                critical_fields = ["title", "description", "country", "source_table", "id"]
                for field in critical_fields:
                    value = tender_dict.get(field, "N/A")
                    value_type = type(value).__name__
                    value_length = len(value) if isinstance(value, str) else "N/A"
                    logger.debug(f"  {field}: {value_type}, length: {value_length}")

                # Save input data for debugging
                debug_dir = "debug_dumps"
                os.makedirs(debug_dir, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                filename = f"{debug_dir}/input_{tender.source_table}_{tender.id}_{timestamp}.json"
                with open(filename, "w", encoding="utf-8") as f:
                    json.dump(tender_dict, f, indent=2, ensure_ascii=False, default=str)
                logger.info(f"Saved debug data to {filename}")

                # Run the LLM agent
                logger.debug(f"Running LLM agent for tender {tender.id}...")
                
                # Execute the agent
                output = await normalizer.agent.run(input_data)
                
                # Log output type for debugging
                output_type = type(output).__name__
                logger.debug(f"Agent returned output of type: {output_type}")
                
                # Save output data for debugging
                if output and hasattr(output, "normalized_data"):
                    output_filename = f"{debug_dir}/output_{tender.source_table}_{tender.id}_{timestamp}.json"
                    with open(output_filename, "w", encoding="utf-8") as f:
                        json.dump(output.normalized_data, f, indent=2, ensure_ascii=False, default=str)
                    logger.info(f"Saved output data to {output_filename}")
                
                return output
            except Exception as e:
                logger.error(f"Error in custom_normalize_with_llm: {str(e)}")
                # Save error data
                error_filename = f"{debug_dir}/error_{tender.source_table}_{tender.id}_{timestamp}.json"
                with open(error_filename, "w", encoding="utf-8") as f:
                    json.dump({
                        "error": str(e),
                        "traceback": traceback.format_exc()
                    }, f, indent=2, ensure_ascii=False)
                logger.info(f"Saved error data to {error_filename}")
                raise
        
        # Call our custom wrapper
        output = await custom_normalize_with_llm()
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Create a result object
        if output and hasattr(output, "normalized_data") and output.normalized_data:
            # Success
            return NormalizationResult(
                tender_id=tender.id,
                source_table=tender.source_table,
                success=True,
                method_used="llm",
                processing_time=processing_time,
                fields_before=len(tender.model_dump()),
                fields_after=len(output.normalized_data),
                improvement_percentage=100.0,  # Placeholder
                normalized_tender=output.normalized_data,
                error=None
            )
        else:
            # Failure
            return NormalizationResult(
                tender_id=tender.id,
                source_table=tender.source_table,
                success=False,
                method_used="llm",
                processing_time=processing_time,
                fields_before=len(tender.model_dump()),
                fields_after=0,
                improvement_percentage=0.0,
                normalized_tender=None,
                error=output.error if hasattr(output, "error") else "Unknown error"
            )
    except Exception as e:
        # Handle exceptions
        processing_time = (datetime.now() - start_time).total_seconds()
        logger.error(f"Error normalizing tender: {str(e)}")
        
        return NormalizationResult(
            tender_id=tender.id,
            source_table=tender.source_table,
            success=False,
            method_used="llm",
            processing_time=processing_time,
            fields_before=len(tender.model_dump()),
            fields_after=0,
            improvement_percentage=0.0,
            normalized_tender=None,
            error=str(e)
        )

async def test_llm_normalization():
    """Run a test of the LLM normalization with diagnostic output."""
    logger.info("Starting LLM normalization test")
    
    # Create test tender
    tender = create_test_tender()
    logger.info(f"Created test tender: {tender.id} from {tender.source_table}")
    
    # Create normalizer
    normalizer = TenderNormalizer()
    # Add logger to normalizer for our test
    normalizer.logger = logger
    logger.info("TenderNormalizer created")
    
    # Normalize tender
    logger.info("Starting normalization process...")
    result = await normalize_tender(normalizer, tender)
    
    # Log results
    if result.success:
        logger.info("✅ SUCCESS: LLM normalization was used")
        logger.info(f"Normalized data: {json.dumps(result.normalized_tender, indent=2)}")
    else:
        logger.error(f"❌ FAILURE: Normalization failed")
        logger.error(f"Error message: {result.error}")
        # Log data from debug files if they exist
        debug_dir = "debug_dumps"
        if os.path.exists(debug_dir):
            for filename in os.listdir(debug_dir):
                if tender.source_table in filename and tender.id in filename:
                    with open(os.path.join(debug_dir, filename), "r") as f:
                        logger.info(f"Contents of {filename}:")
                        logger.info(f.read())
    
    logger.info("LLM test completed")

# Run the test
if __name__ == "__main__":
    if len(sys.argv) > 1:
        os.environ["OPENAI_API_KEY"] = sys.argv[1]
    
    if "OPENAI_API_KEY" not in os.environ or not os.environ["OPENAI_API_KEY"].startswith("sk-"):
        print("⚠️ Please provide an OpenAI API key as an argument: python scripts/test_llm.py YOUR_API_KEY")
        sys.exit(1)
    
    asyncio.run(test_llm_normalization()) 