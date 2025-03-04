"""
Direct implementation of tender normalization using OpenAI API.
This bypasses the pydantic-ai library to avoid the "Expected code to be unreachable" error.
"""
import asyncio
import json
import logging
import os
import time
from datetime import datetime, date
from typing import Any, Dict, List, Optional, Tuple, Union
import traceback

from openai import AsyncOpenAI

from ..config import settings
from ..models.tender import (
    NormalizedTender,
    NormalizationResult,
    RawTender,
    TenderStatus,
    TenderType,
)

logger = logging.getLogger(__name__)

class DirectNormalizer:
    """Direct implementation of tender normalization using OpenAI API."""
    
    def __init__(self, api_key: str, model: str = "gpt-4o", max_retries: int = 3, retry_delay: int = 2):
        """
        Initialize the normalizer with OpenAI API key.

        Args:
            api_key: OpenAI API key
            model: OpenAI model to use, default is gpt-4o
            max_retries: Maximum number of retry attempts for API calls
            retry_delay: Delay between retries in seconds
        """
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.logger = logging.getLogger(__name__)
        
        # Ensure debug directory exists
        os.makedirs("debug_dumps", exist_ok=True)
        
        # Tracking statistics
        self.stats = {
            "total_processed": 0,
            "llm_used": 0,
            "fallback_used": 0,
            "by_source": {},
            "processing_time": [],
        }
        
        # Create system prompt
        self.system_prompt = self._generate_system_prompt()
    
    def _generate_system_prompt(self) -> str:
        """Generate the system prompt for the LLM."""
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
        9. Handle all special characters appropriately, ensuring they don't cause parsing issues.
        10. For nested JSON data, thoroughly explore all levels to extract relevant information.
        
        For each field, always choose the most specific and accurate value from the raw data.
        If a field is not available in the raw data, do not include it in the output.
        
        For date fields, provide standardized ISO format (YYYY-MM-DD) when possible.
        For status fields, normalize to one of: active, closed, awarded, canceled, upcoming, unknown.
        For tender_type fields, normalize to one of: goods, services, works, consulting, mixed, other, unknown.
        
        Your response should follow the exact structure expected, with proper field types and values.
        """
    
    def _generate_user_prompt(self, tender_data: Dict[str, Any]) -> str:
        """
        Generate the user prompt with the tender data.
        
        Args:
            tender_data: Raw tender data to be normalized
            
        Returns:
            Formatted user prompt string
        """
        # Sanitize tender data for JSON serialization
        sanitized_data = self._sanitize_for_json(tender_data)
        
        # Format the JSON data with indentation for better readability
        formatted_json = json.dumps(sanitized_data, indent=2, ensure_ascii=False)
        
        return f"""
        Please normalize the following tender data:
        
        ```json
        {formatted_json}
        ```
        
        Return the normalized data as a JSON object with the following structure:
        
        ```json
        {{
          "tender": {{
            "title": "Normalized title",
            "source_table": "source_name",
            "source_id": "source_specific_id",
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
        Ensure your response is valid JSON. If there are any special characters, handle them correctly.
        """
    
    def _sanitize_for_json(self, data: Any) -> Any:
        """
        Recursively sanitize data to ensure it can be serialized to JSON.
        
        Args:
            data: Data to sanitize
            
        Returns:
            Sanitized data that can be serialized to JSON
        """
        if data is None:
            return None
        elif isinstance(data, (str, int, float, bool)):
            if isinstance(data, str):
                # Replace null bytes and other problematic characters
                return data.replace('\u0000', '')
            return data
        elif isinstance(data, (list, tuple)):
            return [self._sanitize_for_json(item) for item in data]
        elif isinstance(data, dict):
            return {str(k): self._sanitize_for_json(v) for k, v in data.items()}
        elif isinstance(data, (datetime, date)):
            return data.isoformat()
        else:
            # Convert other types to strings
            try:
                return str(data)
            except Exception as e:
                self.logger.warning(f"Failed to sanitize value of type {type(data)}: {e}")
                return f"[Unsupported type: {type(data).__name__}]"
    
    def _save_debug_data(self, 
                         data: Any, 
                         prefix: str, 
                         source: str, 
                         tender_id: str, 
                         timestamp: Optional[str] = None) -> str:
        """
        Save debug data to a file for later analysis.
        
        Args:
            data: Data to save
            prefix: File prefix
            source: Tender source
            tender_id: Tender ID
            timestamp: Optional timestamp, if not provided current time will be used
            
        Returns:
            Path to the saved file
        """
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        
        filename = f"debug_dumps/{prefix}_{source}_{tender_id}_{timestamp}.json"
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                
            self.logger.info(f"Saved debug data to {filename}")
            return filename
        except Exception as e:
            self.logger.error(f"Failed to save debug data: {e}")
            return "failed_to_save"

    async def normalize_tender(self, 
                              tender: Dict[str, Any], 
                              save_debug: bool = True) -> Tuple[Dict[str, Any], str, float]:
        """
        Normalize a single tender using the OpenAI API.
        
        Args:
            tender: Raw tender data
            save_debug: Whether to save debug data
            
        Returns:
            Tuple of (normalized tender data, method used, processing time in seconds)
        """
        tender_id = tender.get('id', 'unknown')
        source = tender.get('source_table', 'unknown')
        
        self.logger.info(f"Normalizing tender {tender_id} from {source}")
        
        # Track statistics
        self.stats["total_processed"] += 1
        if source not in self.stats["by_source"]:
            self.stats["by_source"][source] = {"success": 0, "total": 0}
        self.stats["by_source"][source]["total"] += 1
        
        start_time = time.time()
        
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        
        # Save input data for debugging
        if save_debug:
            self._save_debug_data(tender, "input", source, tender_id, timestamp)
        
        # Create messages
        messages: List[ChatCompletionMessageParam] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self._generate_user_prompt(tender)}
        ]
        
        # Save messages for debugging
        if save_debug:
            self._save_debug_data(messages, "messages", source, tender_id, timestamp)
        
        # Try to normalize using OpenAI API with retries
        for attempt in range(self.max_retries):
            try:
                self.logger.info(f"Calling OpenAI API for tender {tender_id}... (Attempt {attempt+1}/{self.max_retries})")
                
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.1,  # Low temperature for consistent results
                    max_tokens=4000,  # Adjust based on your needs
                    response_format={"type": "json_object"}
                )
                
                # Extract the response content
                response_content = response.choices[0].message.content
                if not response_content:
                    raise ValueError("Empty response from OpenAI API")
                
                # Parse the response JSON
                normalized_data = json.loads(response_content)
                
                # Track success
                self.stats["llm_used"] += 1
                self.stats["by_source"][source]["success"] += 1
                
                # Calculate processing time
                processing_time = time.time() - start_time
                self.stats["processing_time"].append(processing_time)
                
                return normalized_data, "llm", processing_time
                
            except Exception as e:
                error_message = str(e)
                tb = traceback.format_exc()
                error_data = {
                    "error": error_message,
                    "traceback": tb
                }
                
                if save_debug:
                    self._save_debug_data(error_data, "error", source, tender_id, timestamp)
                
                self.logger.error(f"Error normalizing tender {tender_id}: {error_message}")
                
                # If it's the last attempt, raise the error
                if attempt == self.max_retries - 1:
                    self.stats["fallback_used"] += 1
                    
                    # Calculate processing time
                    processing_time = time.time() - start_time
                    self.stats["processing_time"].append(processing_time)
                    
                    # Return empty normalized data as fallback
                    fallback_data = {
                        "tender": {
                            "title": tender.get("title", "Unknown"),
                            "source_table": source,
                            "source_id": tender_id
                        },
                        "missing_fields": ["All fields except title, source_table, source_id"],
                        "notes": f"Normalization failed: {str(e)}"
                    }
                    
                    return fallback_data, "fallback", processing_time
                else:
                    # Wait before retrying
                    await asyncio.sleep(self.retry_delay * (attempt + 1))  # Exponential backoff
    
    async def normalize_tenders(self, 
                              tenders: List[Dict[str, Any]], 
                              save_debug: bool = True,
                              concurrency: int = 5) -> List[Dict[str, Any]]:
        """
        Normalize a list of tenders in parallel.
        
        Args:
            tenders: List of raw tender data
            save_debug: Whether to save debug data
            concurrency: Maximum number of concurrent requests
            
        Returns:
            List of normalized tender data
        """
        self.logger.info(f"Starting normalization of {len(tenders)} tenders with concurrency {concurrency}")
        
        # Create a semaphore to limit concurrency
        semaphore = asyncio.Semaphore(concurrency)
        
        async def _normalize_with_semaphore(tender):
            async with semaphore:
                normalized, method, time_taken = await self.normalize_tender(tender, save_debug)
                return {
                    "tender_id": tender.get("id", "unknown"),
                    "source": tender.get("source_table", "unknown"),
                    "normalized": normalized,
                    "method": method,
                    "time_taken": time_taken
                }
        
        # Process tenders in parallel with limited concurrency
        tasks = [_normalize_with_semaphore(tender) for tender in tenders]
        results = await asyncio.gather(*tasks)
        
        # Log performance statistics
        self._log_performance_stats()
        
        return results
    
    def _log_performance_stats(self):
        """Log performance statistics after processing tenders."""
        total = self.stats["total_processed"]
        llm_used = self.stats["llm_used"]
        fallback_used = self.stats["fallback_used"]
        
        success_rate = (llm_used / total) * 100 if total > 0 else 0
        avg_time = sum(self.stats["processing_time"]) / len(self.stats["processing_time"]) if self.stats["processing_time"] else 0
        
        self.logger.info("=== Performance Statistics ===")
        self.logger.info(f"Total processed: {total}")
        self.logger.info(f"LLM used: {llm_used}")
        self.logger.info(f"Fallback used: {fallback_used}")
        self.logger.info(f"Success rate: {success_rate:.2f}%")
        self.logger.info(f"Average processing time: {avg_time:.2f}s")
        
        if self.stats["by_source"]:
            self.logger.info("=== By Source ===")
            for source, counts in self.stats["by_source"].items():
                source_success_rate = (counts["success"] / counts["total"]) * 100 if counts["total"] > 0 else 0
                self.logger.info(f"{source}: {counts['success']}/{counts['total']} ({source_success_rate:.2f}%)")

# Initialize the normalizer
# Note: Removed direct_normalizer instance initialization since we now require passing an API key 