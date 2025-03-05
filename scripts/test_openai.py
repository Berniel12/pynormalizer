#!/usr/bin/env python3
"""
Simple script to test the OpenAI API directly for tender normalization.
"""
import asyncio
import json
import os
import sys
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional

print("Starting script...")

try:
    from openai import AsyncOpenAI
    print("Successfully imported AsyncOpenAI")
except ImportError as e:
    print(f"Error importing AsyncOpenAI: {e}")
    print("Installing OpenAI package...")
    os.system("pip install openai")
    from openai import AsyncOpenAI
    print("Successfully installed and imported AsyncOpenAI")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("openai_tester")

# Add the parent directory to the path so we can import the src package
sys.path.append(".")
print(f"Python path: {sys.path}")

try:
    from src.models.tender import RawTender
    print("Successfully imported RawTender")
except ImportError as e:
    print(f"Error importing RawTender: {e}")

def create_test_tender(source: str = "sam_gov") -> Dict[str, Any]:
    """
    Create a test tender with potentially problematic data.
    
    Args:
        source: The source table to simulate
        
    Returns:
        A dictionary with tender data
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
    }
    
    return tender_data

def get_system_prompt() -> str:
    """Get the system prompt for the LLM."""
    return """
    You are a specialized AI assistant for tender normalization.
    
    Your task is to normalize tender data from different sources into a consistent format.
    You will be given raw tender data and your goal is to extract and standardize the following fields:
    
    Required fields:
    - title: Short, descriptive title of the tender opportunity (in proper title case, not all caps)
    - source_table: The original data source (e.g., "sam_gov", "wb", "adb", etc.)
    - source_id: Original identifier of the tender in the source system
    
    Important fields to extract (if available):
    - description: Detailed description of what the tender requires
    - tender_type: Type of tender (goods, services, works, consulting, mixed, other, unknown)
    - status: Current status (active, closed, awarded, canceled, upcoming, unknown)
    - publication_date: When the tender was published
    - deadline_date: Submission deadline
    - country: Country where the work/services will be performed
    - city: Specific city or location
    - organization_name: Name of the organization issuing the tender
    - organization_id: ID of the issuing organization
    - buyer: Entity making the purchase (often same as organization_name)
    - project_name: Name of the overall project
    - project_id: ID of the project
    - project_number: Reference number for the project
    - sector: Business/industry sector
    - estimated_value: Monetary value of the tender
    - currency: Currency of the tender value
    - contact information:
      - contact_name: Name of the contact person
      - contact_email: Email address for inquiries
      - contact_phone: Phone number for inquiries
      - contact_address: Physical contact address
    - url: Main URL of the tender notice
    - document_links: Links to tender documents
    - language: Original language of the tender
    - notice_id: ID of the specific notice
    - reference_number: Reference number for the tender
    - procurement_method: Method used for procurement
    
    Guidelines for normalization:
    1. Extract ALL available fields from the raw data, even if they seem redundant.
    2. Format titles in proper Title Case, not ALL CAPS. Preserve recognized acronyms.
    3. Clean up excessive numbers and codes from titles while preserving essential information.
    4. Remove excessive punctuation and special characters from all text fields.
    5. If text appears to be in a language other than English, note this in the 'language' field.
    6. EXTREMELY IMPORTANT: Look for URLs in ALL fields, especially description fields and source_data. Extract any URLs found and include them in the url field.
    7. EXTREMELY IMPORTANT: Look for contact information (email, phone, address) throughout the data and make sure to extract it.
    8. Check both direct fields and nested fields in source_data for all relevant information.
    
    For each field, always choose the most specific and accurate value from the raw data.
    If a field is not available in the raw data, do not include it in the output.
    
    For date fields, provide standardized ISO format (YYYY-MM-DD) when possible.
    For status fields, normalize to one of: active, closed, awarded, canceled, upcoming, unknown.
    For tender_type fields, normalize to one of: goods, services, works, consulting, mixed, other, unknown.
    
    Your response should follow the exact structure expected, with proper field types and values.
    """

def get_user_prompt(tender_data: Dict[str, Any]) -> str:
    """
    Create a user prompt for the LLM.
    
    Args:
        tender_data: The tender data to normalize
        
    Returns:
        A string with the user prompt
    """
    return f"""
    Please normalize the following tender data:
    
    ```json
    {json.dumps(tender_data, indent=2, ensure_ascii=False)}
    ```
    
    Return the normalized data as a JSON object with the following structure:
    
    ```json
    {{
      "tender": {{
        "title": "Normalized title",
        "source_table": "{tender_data.get('source_table', 'unknown')}",
        "source_id": "{tender_data.get('id', 'unknown')}",
        "description": "Normalized description",
        "tender_type": "goods|services|works|consulting|mixed|other|unknown",
        "status": "active|closed|awarded|canceled|upcoming|unknown",
        "publication_date": "YYYY-MM-DD",
        "deadline_date": "YYYY-MM-DD",
        "country": "Country name",
        "city": "City name",
        "organization_name": "Organization name",
        "organization_id": "Organization ID",
        "buyer": "Buyer name",
        "project_name": "Project name",
        "project_id": "Project ID",
        "project_number": "Project number",
        "sector": "Sector",
        "estimated_value": 1000.00,
        "currency": "USD",
        "contact_name": "Contact name",
        "contact_email": "contact@example.com",
        "contact_phone": "123-456-7890",
        "contact_address": "Contact address",
        "url": "https://example.com",
        "document_links": ["https://example.com/doc1", "https://example.com/doc2"],
        "language": "en",
        "notice_id": "Notice ID",
        "reference_number": "Reference number",
        "procurement_method": "Procurement method"
      }},
      "missing_fields": ["field1", "field2"],
      "notes": "Any notes about the normalization process"
    }}
    ```
    
    Only include fields that are available in the raw data or can be inferred from it.
    """

async def test_openai_normalization():
    """Test the OpenAI API directly for tender normalization."""
    print("Starting OpenAI normalization test")
    logger.info("Starting OpenAI normalization test")
    
    # Check if we have the required environment variables
    if "OPENAI_API_KEY" not in os.environ or not os.environ["OPENAI_API_KEY"].startswith("sk-"):
        print("OPENAI_API_KEY environment variable is not set or invalid")
        logger.error("OPENAI_API_KEY environment variable is not set or invalid")
        return
    
    # Create test tender
    tender_data = create_test_tender()
    print(f"Created test tender: {tender_data['id']} from {tender_data['source_table']}")
    logger.info(f"Created test tender: {tender_data['id']} from {tender_data['source_table']}")
    
    # Save input data for debugging
    debug_dir = "debug_dumps"
    os.makedirs(debug_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    input_filename = f"{debug_dir}/openai_input_{tender_data['source_table']}_{tender_data['id']}_{timestamp}.json"
    with open(input_filename, "w", encoding="utf-8") as f:
        json.dump(tender_data, f, indent=2, ensure_ascii=False)
    print(f"Saved input data to {input_filename}")
    logger.info(f"Saved input data to {input_filename}")
    
    # Create OpenAI client
    client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
    print("Created OpenAI client")
    logger.info("Created OpenAI client")
    
    # Create messages
    messages = [
        {"role": "system", "content": get_system_prompt()},
        {"role": "user", "content": get_user_prompt(tender_data)}
    ]
    
    # Save messages for debugging
    messages_filename = f"{debug_dir}/openai_messages_{tender_data['source_table']}_{tender_data['id']}_{timestamp}.json"
    with open(messages_filename, "w", encoding="utf-8") as f:
        json.dump(messages, f, indent=2, ensure_ascii=False)
    print(f"Saved messages to {messages_filename}")
    logger.info(f"Saved messages to {messages_filename}")
    
    try:
        # Call OpenAI API
        print("Calling OpenAI API...")
        logger.info("Calling OpenAI API...")
        start_time = datetime.now()
        
        response = client.chat.completions.create(
            messages=messages,
            model="gpt-4o-mini",  # Use the appropriate model
            temperature=0.0,  # Use deterministic output
            max_tokens=4000,  # Adjust as needed
        )
        
        processing_time = (datetime.now() - start_time).total_seconds()
        print(f"OpenAI API call completed in {processing_time:.2f} seconds")
        logger.info(f"OpenAI API call completed in {processing_time:.2f} seconds")
        
        # Save response for debugging
        response_dict = {
            "id": response.id,
            "model": response.model,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            },
            "content": response.choices[0].message.content
        }
        
        response_filename = f"{debug_dir}/openai_response_{tender_data['source_table']}_{tender_data['id']}_{timestamp}.json"
        with open(response_filename, "w", encoding="utf-8") as f:
            json.dump(response_dict, f, indent=2, ensure_ascii=False)
        print(f"Saved response to {response_filename}")
        logger.info(f"Saved response to {response_filename}")
        
        # Parse the response
        try:
            # Extract JSON from the response
            content = response.choices[0].message.content
            # Find JSON block
            json_start = content.find("{")
            json_end = content.rfind("}")
            
            if json_start >= 0 and json_end >= 0:
                json_str = content[json_start:json_end+1]
                normalized_data = json.loads(json_str)
                
                # Save normalized data
                normalized_filename = f"{debug_dir}/openai_normalized_{tender_data['source_table']}_{tender_data['id']}_{timestamp}.json"
                with open(normalized_filename, "w", encoding="utf-8") as f:
                    json.dump(normalized_data, f, indent=2, ensure_ascii=False)
                print(f"Saved normalized data to {normalized_filename}")
                logger.info(f"Saved normalized data to {normalized_filename}")
                
                # Print summary
                print("✅ SUCCESS: OpenAI normalization completed")
                logger.info("✅ SUCCESS: OpenAI normalization completed")
                if "tender" in normalized_data:
                    tender = normalized_data["tender"]
                    print(f"Normalized title: {tender.get('title', 'N/A')}")
                    print(f"Normalized tender_type: {tender.get('tender_type', 'N/A')}")
                    print(f"Normalized status: {tender.get('status', 'N/A')}")
                    print(f"Fields normalized: {len(tender)}")
                    logger.info(f"Normalized title: {tender.get('title', 'N/A')}")
                    logger.info(f"Normalized tender_type: {tender.get('tender_type', 'N/A')}")
                    logger.info(f"Normalized status: {tender.get('status', 'N/A')}")
                    logger.info(f"Fields normalized: {len(tender)}")
                
                if "missing_fields" in normalized_data:
                    print(f"Missing fields: {normalized_data['missing_fields']}")
                    logger.info(f"Missing fields: {normalized_data['missing_fields']}")
                
                if "notes" in normalized_data:
                    print(f"Notes: {normalized_data['notes']}")
                    logger.info(f"Notes: {normalized_data['notes']}")
            else:
                print("Failed to extract JSON from response")
                print(f"Response content: {content}")
                logger.error("Failed to extract JSON from response")
                logger.info(f"Response content: {content}")
        except Exception as e:
            print(f"Error parsing response: {str(e)}")
            print(f"Response content: {response.choices[0].message.content}")
            logger.error(f"Error parsing response: {str(e)}")
            logger.info(f"Response content: {response.choices[0].message.content}")
    
    except Exception as e:
        print(f"Error calling OpenAI API: {str(e)}")
        logger.error(f"Error calling OpenAI API: {str(e)}")
    
    print("OpenAI test completed")
    logger.info("OpenAI test completed")

if __name__ == "__main__":
    print("Script main block started")
    # Set API key from command line if provided
    if len(sys.argv) > 1:
        os.environ["OPENAI_API_KEY"] = sys.argv[1]
        print(f"Set API key from command line argument")
    
    print("Running asyncio.run(test_openai_normalization())")
    asyncio.run(test_openai_normalization())
    print("Script completed") 