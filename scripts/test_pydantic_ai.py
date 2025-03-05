#!/usr/bin/env python3
"""
Script to test the pydantic-ai library directly.
"""
import os
import sys
import json
import asyncio
from datetime import datetime
from typing import Dict, Any, List, Optional

print("Starting pydantic-ai test...")

# Try to install the pydantic-ai package if not already installed
try:
    from pydantic import BaseModel, Field
    from pydantic_ai import Agent
    print("pydantic-ai package is already installed")
except ImportError:
    print("Installing pydantic-ai package...")
    os.system("pip install pydantic-ai")
    from pydantic import BaseModel, Field
    from pydantic_ai import Agent
    print("pydantic-ai package installed successfully")

# Check if API key is provided
if len(sys.argv) > 1:
    os.environ["OPENAI_API_KEY"] = sys.argv[1]
    print("Using API key from command line argument")
elif "OPENAI_API_KEY" in os.environ:
    print("Using API key from environment variable")
else:
    print("Error: No API key provided")
    print("Usage: python test_pydantic_ai.py YOUR_API_KEY")
    sys.exit(1)

# Define input and output models
class NormalizationInput(BaseModel):
    """Input for the LLM-based tender normalization."""
    raw_tender: Dict[str, Any] = Field(
        ..., description="The raw tender data to be normalized"
    )
    source_table: str = Field(
        ..., description="Source table name (e.g. 'sam_gov', 'wb', etc.)"
    )

class NormalizationOutput(BaseModel):
    """Output from the LLM-based tender normalization."""
    tender: Dict[str, Any] = Field(
        ..., description="The normalized tender data"
    )
    missing_fields: List[str] = Field(
        default_factory=list, description="Fields that could not be normalized"
    )
    notes: Optional[str] = Field(
        None, description="Any notes or explanations about the normalization process"
    )

# Create a test tender
def create_test_tender(source: str = "sam_gov") -> Dict[str, Any]:
    """Create a test tender."""
    current_time = datetime.now().isoformat()
    
    return {
        "id": f"test-{source}-001",
        "source_table": source,
        "title": "Test tender with special chars",
        "description": "This is a test tender description.",
        "publication_date": current_time,
        "country": "United States",
        "organization_name": "Test Organization",
        "url": "https://example.com/tender/123",
    }

# Define the system prompt
def get_system_prompt() -> str:
    """Get the system prompt for the LLM."""
    return """
    You are a specialized AI assistant for tender normalization.
    
    Your task is to normalize tender data from different sources into a consistent format.
    You will be given raw tender data and your goal is to extract and standardize fields.
    
    Return the normalized data in the expected format.
    """

async def test_pydantic_ai():
    """Test the pydantic-ai library."""
    print("Creating Agent...")
    
    # Create the agent
    agent = Agent(
        model="gpt-4o-mini",
        result_type=NormalizationOutput,
        system_prompt=get_system_prompt(),
    )
    
    print("Agent created")
    
    # Create input data
    tender_data = create_test_tender()
    input_data = NormalizationInput(
        raw_tender=tender_data,
        source_table=tender_data["source_table"]
    )
    
    print("Input data created")
    
    # Save input data for debugging
    debug_dir = "debug_dumps"
    os.makedirs(debug_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    input_filename = f"{debug_dir}/pydantic_ai_input_{timestamp}.json"
    
    with open(input_filename, "w") as f:
        json.dump(input_data.model_dump(), f, indent=2)
    
    print(f"Input data saved to {input_filename}")
    
    # Run the agent
    print("Running agent...")
    try:
        output = await agent.run(input_data)
        
        print("\nAgent run successful!")
        print(f"Output type: {type(output).__name__}")
        
        # Save output data
        output_filename = f"{debug_dir}/pydantic_ai_output_{timestamp}.json"
        
        with open(output_filename, "w") as f:
            json.dump(output.model_dump(), f, indent=2)
        
        print(f"Output data saved to {output_filename}")
        
        # Print summary
        print("\nNormalization results:")
        print(f"Tender fields: {len(output.tender)}")
        print(f"Missing fields: {output.missing_fields}")
        if output.notes:
            print(f"Notes: {output.notes}")
        
    except Exception as e:
        print(f"Error running agent: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_pydantic_ai()) 