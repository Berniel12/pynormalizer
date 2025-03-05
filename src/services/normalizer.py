"""
Tender normalization service using PydanticAI.
"""
import asyncio
import json
import logging
import os
import re
import string
import time
from datetime import datetime, date
from typing import Any, Dict, List, Optional, TypeVar, Union
import pprint
import unicodedata
import sys
import traceback
import json as json_module
import openai
from pathlib import Path

from langdetect import detect, LangDetectException
from deep_translator import GoogleTranslator
from pydantic import BaseModel, Field, ValidationError, model_validator
from pydantic_ai import Agent, RunContext

from ..config import normalizer_config, settings, NormalizerConfig
from ..models.tender import (
    NormalizedTender,
    NormalizationResult,
    RawTender,
    TenderStatus,
    TenderType,
)

logger = logging.getLogger(__name__)

# Set up a dedicated logger for quality control
quality_logger = logging.getLogger("tender_quality")
quality_handler = logging.FileHandler("tender_quality.log")
quality_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
quality_logger.addHandler(quality_handler)
quality_logger.setLevel(logging.INFO)
quality_logger.propagate = False  # Don't propagate to root logger

T = TypeVar("T", bound=BaseModel)

# Language detection and translation functions
def detect_language(text: str) -> str:
    """Detect the language of a text string."""
    if not text or not isinstance(text, str) or len(text.strip()) < 10:
        return "en"  # Default to English for very short or empty text
    
    try:
        return detect(text)
    except LangDetectException:
        return "en"  # Default to English if detection fails

def translate_to_english(text: str, source_lang: str = "auto") -> str:
    """Translate text to English if it's not already in English."""
    if not text or not isinstance(text, str) or len(text.strip()) < 5:
        return text
    
    if source_lang == "en":
        return text
    
    try:
        translator = GoogleTranslator(source=source_lang, target='en')
        return translator.translate(text)
    except Exception as e:
        logger.error(f"Translation error: {str(e)}")
        return text  # Return original text if translation fails

def format_title(title: str) -> str:
    """
    Format a title to be more readable:
    - Convert ALL CAPS to Title Case
    - Clean up excessive punctuation and whitespace
    - Preserve common acronyms
    """
    if not title or not isinstance(title, str):
        return title
    
    # Define common acronyms to preserve
    common_acronyms = ["UN", "EU", "US", "UK", "IT", "ICT", "USD", "EUR", "GBP", "VAT", "RFP", "RFQ"]
    acronyms_pattern = '|'.join(r'\b' + re.escape(acr) + r'\b' for acr in common_acronyms)
    
    # Check if title is ALL CAPS or mostly caps (>70% uppercase)
    uppercase_ratio = sum(1 for c in title if c.isupper() and c.isalpha()) / max(1, sum(1 for c in title if c.isalpha()))
    
    if uppercase_ratio > 0.7:
        # Replace acronyms with placeholders
        placeholder_map = {}
        if acronyms_pattern:
            def replace_with_placeholder(match):
                placeholder = f"__ACRONYM_{len(placeholder_map)}__"
                placeholder_map[placeholder] = match.group(0)
                return placeholder
            
            title = re.sub(acronyms_pattern, replace_with_placeholder, title)
        
        # Convert to title case
        title = string.capwords(title.lower())
        
        # Restore acronyms
        for placeholder, acronym in placeholder_map.items():
            title = title.replace(placeholder, acronym)
    
    # Clean up excessive punctuation and whitespace
    title = re.sub(r'\s+', ' ', title)  # Remove multiple spaces
    title = re.sub(r'[^\w\s\-\.,:()/]', '', title)  # Remove unusual symbols
    title = re.sub(r'[\-_]{2,}', '-', title)  # Replace multiple hyphens with single
    
    return title.strip()

def extract_urls_from_text(text: str) -> List[str]:
    """
    Extract URLs from text content.
    
    Args:
        text: The text to search for URLs
        
    Returns:
        List of URLs found in the text
    """
    if not text or not isinstance(text, str):
        return []
    
    # Regular expression to find URLs in text
    url_pattern = re.compile(
        r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    )
    
    # Find all matches
    urls = url_pattern.findall(text)
    
    # Clean up URLs (remove trailing punctuation)
    clean_urls = []
    for url in urls:
        # Remove trailing punctuation
        if url and url[-1] in '.,;:)]}':
            url = url[:-1]
        if url:
            clean_urls.append(url)
    
    return clean_urls

def extract_emails_from_text(text: str) -> List[str]:
    """
    Extract email addresses from text content.
    
    Args:
        text: The text to search for email addresses
        
    Returns:
        List of email addresses found in the text
    """
    if not text or not isinstance(text, str):
        return []
    
    # Regular expression to find email addresses in text
    email_pattern = re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}')
    
    # Find all matches
    emails = email_pattern.findall(text)
    
    return emails

def log_quality_check(tender_id: str, source_table: str, raw_data: Dict[str, Any], normalized_data: Dict[str, Any]) -> None:
    """
    Log detailed information about the normalization process for quality checking.
    
    Args:
        tender_id: The ID of the tender
        source_table: The source table of the tender
        raw_data: The original raw data
        normalized_data: The normalized data
    """
    # Create a formatted comparison for logging
    quality_data = {
        "tender_id": tender_id,
        "source_table": source_table,
        "timestamp": datetime.utcnow().isoformat(),
        "raw_fields_count": len(raw_data) if raw_data else 0,
        "normalized_fields_count": len(normalized_data) if normalized_data else 0,
        "missing_critical_fields": [],
        "field_comparison": {}
    }
    
    # Define critical fields to check
    critical_fields = [
        "title", "description", "url", "contact_name", "contact_email",
        "contact_phone", "publication_date", "deadline_date"
    ]
    
    # Check for missing critical fields
    for field in critical_fields:
        if field not in normalized_data or not normalized_data.get(field):
            quality_data["missing_critical_fields"].append(field)
    
    # Add field comparison
    for field in normalized_data:
        quality_data["field_comparison"][field] = {
            "normalized_value": normalized_data.get(field),
            "found_in_raw": field in raw_data,
            "raw_value": raw_data.get(field) if field in raw_data else None
        }
    
    # Log the quality data
    quality_logger.info(f"QUALITY CHECK FOR {source_table}:{tender_id}\n" + 
                      pprint.pformat(quality_data, width=120, compact=False))

class NormalizationInput(BaseModel):
    """Input for the LLM-based tender normalization."""
    id: str = Field(..., description="Unique identifier for the tender")
    source_table: str = Field(..., description="Source table name (e.g. 'sam_gov', 'wb', etc.)")
    title: Optional[str] = Field(None, description="Tender title")
    description: Optional[str] = Field(None, description="Tender description")
    publication_date: Optional[str] = Field(None, description="Publication date (ISO format)")
    deadline_date: Optional[str] = Field(None, description="Deadline date (ISO format)")
    country: Optional[str] = Field(None, description="Country")
    organization_name: Optional[str] = Field(None, description="Organization name")
    source_data: Optional[Dict[str, Any]] = Field(None, description="Source-specific data (nested structure)")
    raw_tender: Dict[str, Any] = Field(
        default_factory=dict, 
        description="The complete raw tender data - will be converted to simplified structure"
    )

    def model_dump(self, *args, **kwargs):
        """
        Override model_dump to ensure that raw_tender is serialized properly for the LLM.
        We need to remove any problematic characters and data types that might cause issues.
        """
        data = super().model_dump(*args, **kwargs)
        
        # Deep sanitize the raw_tender data to avoid serialization issues
        if "raw_tender" in data and data["raw_tender"]:
            # Convert raw_tender to a simplified string-based structure
            # This avoids PydanticAI serialization issues while still providing the data
            simplified = {}
            for key, value in data["raw_tender"].items():
                # Skip empty values
                if value is None or value == "":
                    continue
                
                # Convert complex nested objects to strings
                if isinstance(value, dict):
                    simplified[key] = str(value)
                elif isinstance(value, list):
                    simplified[key] = str(value)
                else:
                    # Keep simple values as is
                    simplified[key] = value
                    
            data["raw_tender"] = simplified
            
        return data


class NormalizationOutput(BaseModel):
    """
    Output from the LLM-based tender normalization.
    """
    
    tender: Dict[str, Any] = Field(
        ..., description="The normalized tender data"
    )
    missing_fields: List[str] = Field(
        default_factory=list, description="Fields that could not be normalized"
    )
    notes: Optional[str] = Field(
        None, description="Any notes or explanations about the normalization process"
    )
    
    # Field validation
    @model_validator(mode="after")
    def validate_tender_fields(self) -> "NormalizationOutput":
        """Validate that the tender has required fields and correct types."""
        required_fields = ["title", "source_table", "source_id"]
        for field in required_fields:
            if field not in self.tender:
                raise ValueError(f"Missing required field: {field}")
        
        # Validate field types
        tender = self.tender
        
        # Date fields
        date_fields = ["publication_date", "deadline_date", "normalized_at"]
        for field in date_fields:
            if field in tender and tender[field]:
                if tender[field] in ["Unknown", "unknown"]:
                    tender[field] = None
                    continue
                
                if not isinstance(tender[field], (str, datetime, date)):
                    raise ValueError(f"Invalid type for {field}: must be a date/datetime or ISO string")
                
                # Try to parse the date if it's a string
                if isinstance(tender[field], str):
                    try:
                        from dateutil import parser
                        parsed_date = parser.parse(tender[field]).date().isoformat()
                        tender[field] = parsed_date
                    except Exception:
                        # If we can't parse it, set to None
                        tender[field] = None
        
        # Numeric fields
        numeric_fields = ["estimated_value"]
        for field in numeric_fields:
            if field in tender and tender[field] is not None:
                try:
                    # Convert to float if it's a string
                    if isinstance(tender[field], str):
                        tender[field] = float(tender[field].replace(",", ""))
                    # Ensure it's a number
                    float(tender[field])
                except (ValueError, TypeError):
                    raise ValueError(f"Invalid numeric value for {field}")
        
        # Status field
        if "status" in tender and tender["status"]:
            valid_statuses = ["active", "closed", "awarded", "canceled", "upcoming", "unknown"]
            if isinstance(tender["status"], str) and tender["status"].lower() not in valid_statuses:
                tender["status"] = "unknown"
        
        # Tender type field
        if "tender_type" in tender and tender["tender_type"]:
            valid_types = ["goods", "services", "works", "consulting", "mixed", "other", "unknown"]
            if isinstance(tender["tender_type"], str) and tender["tender_type"].lower() not in valid_types:
                tender["tender_type"] = "unknown"
        
        return self


class TenderNormalizer:
    """Service for normalizing tender data using PydanticAI."""

    def __init__(self):
        """Initialize the tender normalizer."""
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing TenderNormalizer")
        
        # Add config attribute
        self.config = NormalizerConfig()
        
        # Set up OpenAI API key
        if not os.environ.get("OPENAI_API_KEY"):
            self.logger.info("Setting OpenAI API key from config")
            if "OPENAI_KEY" in os.environ:
                os.environ["OPENAI_API_KEY"] = os.environ["OPENAI_KEY"]
            elif settings.openai_api_key.get_secret_value():
                os.environ["OPENAI_API_KEY"] = settings.openai_api_key.get_secret_value()
                
        # Set up the agent for normalization with system_prompt
        try:
            self.logger.info(f"Creating PydanticAI agent with model: {settings.openai_model}")
            self.agent = Agent(
                model=settings.openai_model,
                result_type=NormalizationOutput,
                system_prompt=self._get_system_prompt(),
            )
            self.logger.info("PydanticAI agent created successfully")
        except Exception as e:
            self.logger.error(f"Failed to create PydanticAI agent: {str(e)}")
            self.logger.error(traceback.format_exc())
            # We'll continue initialization but the agent will be created on first use
            self.agent = None
        
        # Performance tracking
        self.performance_stats = {
            "total_processed": 0,
            "llm_used": 0,
            "fallback_used": 0,
            "success_rate": 0,
            "avg_processing_time": 0,
            "total_processing_time": 0,
            "by_source": {},
        }

    def _get_system_prompt(self) -> str:
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

    def _should_use_llm(self, tender: RawTender) -> tuple[bool, str]:
        """
        Determine if LLM normalization should be used based on tender contents.
        
        Args:
            tender: The raw tender data
            
        Returns:
            Tuple of (should_use_llm, reason)
        """
        # Check source-specific configuration
        source_setting = normalizer_config.use_llm_for_sources.get(
            tender.source_table,
            normalizer_config.use_llm_for_sources.get("_default", True)
        )
        
        if not source_setting:
            return False, f"LLM disabled for source: {tender.source_table}"
        
        # Check for missing critical fields - if any are missing, definitely use LLM
        missing_critical = []
        for field in normalizer_config.critical_fields:
            value = getattr(tender, field, None)
            if not value or str(value).strip() == "":
                missing_critical.append(field)
        
        if missing_critical:
            return True, f"Missing critical fields: {', '.join(missing_critical)}"
        
        # Very large tenders might exceed token limits and should use fallback
        if tender.description and len(tender.description) > 15000:
            return False, "Description too long for LLM processing"
        
        # Extremely minimal content tenders
        if (tender.description and len(tender.description) < 50 and 
                tender.title and len(tender.title) < 20):
            return False, "Tender content is minimal, using direct parsing"
        
        # Default to using LLM
        return True, "Default processing path"

    def _save_debug_data(self, data: Any, data_type: str) -> None:
        """
        Save debug data to a file.
        
        Args:
            data: The data to save
            data_type: The type of data (input, output, error, messages)
        """
        if not self.config.save_debug_data:
            return
            
        try:
            # Create debug directory if it doesn't exist
            debug_dir = Path("debug_dumps")
            debug_dir.mkdir(exist_ok=True)
            
            # Get tender ID and source from data if available
            tender_id = None
            source = None
            
            if isinstance(data, dict):
                tender_id = data.get("id")
                source = data.get("source_table")
            elif hasattr(data, "id") and hasattr(data, "source_table"):
                tender_id = data.id
                source = data.source_table
            elif hasattr(data, "normalized_data") and isinstance(data.normalized_data, dict):
                tender_id = data.normalized_data.get("id")
                source = data.normalized_data.get("source_table")
                
            if not tender_id or not source:
                # Use timestamp as fallback
                timestamp = time.strftime("%Y%m%d%H%M%S")
                filename = f"{data_type}_{timestamp}.json"
            else:
                # Create filename with tender ID, source, and timestamp
                timestamp = time.strftime("%Y%m%d%H%M%S")
                filename = f"{data_type}_{source}_{tender_id}_{timestamp}.json"
            
            # Convert data to JSON
            if hasattr(data, "model_dump"):
                json_data = data.model_dump()
            elif isinstance(data, dict):
                json_data = data
            else:
                json_data = {"data": str(data)}
            
            # Save to file
            with open(debug_dir / filename, "w", encoding="utf-8") as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False, default=str)
                
            self.logger.info(f"Saved debug data to {debug_dir / filename}")
        except Exception as e:
            self.logger.error(f"Failed to save debug data: {str(e)}")
            self.logger.error(traceback.format_exc())

    async def _normalize_with_llm(self, input_data: NormalizationInput) -> NormalizationOutput:
        """
        Normalize a tender using the LLM.
        
        Args:
            input_data: The input data for normalization
            
        Returns:
            The normalized output
        """
        tender_id = input_data.id
        source = input_data.source_table
        
        self.logger.info(f"Normalizing tender {tender_id} from {source} with LLM")
        
        # Save debug data if enabled
        if self.config.save_debug_data:
            self._save_debug_data(input_data, "input")
        
        try:
            # Try using PydanticAI first
            self.logger.info(f"Attempting to normalize with PydanticAI")
            output = await self.agent.run(input_data)
            self.logger.info(f"PydanticAI normalization successful")
            
            # Save debug data if enabled
            if self.config.save_debug_data:
                self._save_debug_data(output, "output")
            
            return output
        except Exception as e:
            error_str = str(e)
            self.logger.error(f"PydanticAI normalization failed: {error_str}")
            
            # If the error is related to the "Expected code to be unreachable" issue
            # or any other PydanticAI serialization problem, use direct approach
            if "Expected code to be unreachable" in error_str or "serialization" in error_str.lower():
                self.logger.info(f"Falling back to direct approach for tender {tender_id}")
                
                # Create system prompt
                system_prompt = self._get_system_prompt()
                
                # Create user prompt with the tender data
                user_prompt = f"""
                Please normalize the following tender data:
                
                ```json
                {json.dumps(input_data.model_dump(), indent=2)}
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
                
                # Create messages for OpenAI API
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
                
                # Save messages for debugging
                if self.config.save_debug_data:
                    debug_data = {
                        "messages": messages
                    }
                    self._save_debug_data(debug_data, "messages")
                
                # Call OpenAI API
                self.logger.info(f"Calling OpenAI API directly")
                client = openai.AsyncOpenAI(api_key=settings.openai_api_key.get_secret_value())
                
                try:
                    response = await client.chat.completions.create(
                        model=settings.openai_model,
                        messages=messages,
                        temperature=0.1,
                        max_tokens=4000
                    )
                    
                    # Extract response content
                    response_content = response.choices[0].message.content
                    
                    # Parse JSON from response
                    try:
                        # Extract JSON from the response (it might be wrapped in markdown code blocks)
                        json_match = re.search(r'```json\s*([\s\S]*?)\s*```', response_content)
                        if json_match:
                            json_str = json_match.group(1)
                        else:
                            json_str = response_content
                        
                        # Parse the JSON
                        normalized_data = json.loads(json_str)
                        
                        # Create NormalizationOutput
                        output = NormalizationOutput(
                            normalized_data=normalized_data.get("tender", {}),
                            missing_fields=normalized_data.get("missing_fields", []),
                            notes=normalized_data.get("notes", "Normalized using direct approach"),
                            error=None
                        )
                        
                        # Save debug data if enabled
                        if self.config.save_debug_data:
                            self._save_debug_data(output, "output")
                        
                        return output
                    except json.JSONDecodeError as json_err:
                        self.logger.error(f"Failed to parse JSON from response: {str(json_err)}")
                        self.logger.error(f"Response content: {response_content}")
                        
                        # Create error output
                        output = NormalizationOutput(
                            normalized_data={},
                            missing_fields=[],
                            notes="Failed to parse JSON from response",
                            error=f"JSON parse error: {str(json_err)}"
                        )
                        
                        # Save debug data if enabled
                        if self.config.save_debug_data:
                            self._save_debug_data(output, "error")
                        
                        return output
                except Exception as api_err:
                    self.logger.error(f"OpenAI API call failed: {str(api_err)}")
                    
                    # Create error output
                    output = NormalizationOutput(
                        normalized_data={},
                        missing_fields=[],
                        notes="OpenAI API call failed",
                        error=f"API error: {str(api_err)}"
                    )
                    
                    # Save debug data if enabled
                    if self.config.save_debug_data:
                        error_data = {
                            "error": str(api_err),
                            "traceback": traceback.format_exc()
                        }
                        self._save_debug_data(error_data, "error")
                    
                    return output
            else:
                # Re-raise if not a serialization error
                raise
        except Exception as e:
            self.logger.error(f"Error in _normalize_with_llm: {str(e)}")
            self.logger.error(traceback.format_exc())
            
            # Save debug data if enabled
            if self.config.save_debug_data:
                error_data = {
                    "error": str(e),
                    "traceback": traceback.format_exc()
                }
                self._save_debug_data(error_data, "error")
            
            # Return error output
            return NormalizationOutput(
                normalized_data={},
                missing_fields=[],
                notes="Normalization failed",
                error=str(e)
            )

    def _identify_json_serialization_issues(self, data: Dict[str, Any]) -> None:
        """
        Attempt to identify fields that cause JSON serialization issues.
        
        Args:
            data: The data to check for serialization issues
        """
        self.logger.debug("Identifying JSON serialization issues...")
        
        # Check each top-level field
        for key, value in data.items():
            try:
                json.dumps({key: value})
                self.logger.debug(f"  Field '{key}' serializes OK")
            except Exception as e:
                self.logger.error(f"  Field '{key}' has serialization issue: {str(e)}")
                
                # If it's a dictionary, check each sub-field
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        try:
                            json.dumps({sub_key: sub_value})
                            self.logger.debug(f"    Sub-field '{key}.{sub_key}' serializes OK")
                        except Exception as e:
                            self.logger.error(f"    Sub-field '{key}.{sub_key}' has serialization issue: {str(e)}")
                
                # If it's a list, check each element
                elif isinstance(value, list):
                    for i, item in enumerate(value):
                        try:
                            json.dumps(item)
                            self.logger.debug(f"    List item '{key}[{i}]' serializes OK")
                        except Exception as e:
                            self.logger.error(f"    List item '{key}[{i}]' has serialization issue: {str(e)}")
    
    def _validate_normalized_output(self, output: NormalizationOutput, input_data: NormalizationInput) -> None:
        """
        Validate that the normalized output contains all required fields.
        
        Args:
            output: The normalization output to validate
            input_data: The original input data
        """
        if not output or not hasattr(output, "normalized_data"):
            self.logger.error("Normalized output is missing or invalid")
            return
        
        # Check for essential fields
        normalized_data = output.normalized_data
        essential_fields = ["title", "description", "country", "source_table", "id", "organization_name"]
        
        for field in essential_fields:
            if field not in normalized_data or not normalized_data[field]:
                self.logger.warning(f"Normalized data is missing essential field: {field}")
                
                # Use input value as fallback if available
                raw_tender = input_data.model_dump().get("raw_tender", {})
                if field in raw_tender and raw_tender[field]:
                    normalized_data[field] = raw_tender[field]
            # If tender type contains certain keywords, map accordingly
            tender_type_lower = tender_type.lower()
            if any(word in tender_type_lower for word in ["good", "supply", "product", "equipment"]):
                normalized_data["tender_type"] = "goods"
            elif any(word in tender_type_lower for word in ["service", "maintenance"]):
                normalized_data["tender_type"] = "services"
            elif any(word in tender_type_lower for word in ["work", "construction", "build"]):
                normalized_data["tender_type"] = "works"
            elif any(word in tender_type_lower for word in ["consult", "advisory"]):
                normalized_data["tender_type"] = "consulting"
            else:
                # Default to "other" if we can't determine the type
                normalized_data["tender_type"] = "other"
        else:
            # If no tender type, try to infer from other fields
            description = normalized_data.get("description", "")
            title = normalized_data.get("title", "")
            
            if description or title:
                combined_text = (description + " " + title).lower()
                if any(word in combined_text for word in ["good", "supply", "product", "equipment"]):
                    normalized_data["tender_type"] = "goods"
                elif any(word in combined_text for word in ["service", "maintenance"]):
                    normalized_data["tender_type"] = "services"
                elif any(word in combined_text for word in ["work", "construction", "build"]):
                    normalized_data["tender_type"] = "works"
                elif any(word in combined_text for word in ["consult", "advisory"]):
                    normalized_data["tender_type"] = "consulting"
                else:
                    normalized_data["tender_type"] = "unknown"
            else:
                normalized_data["tender_type"] = "unknown"

    def _ensure_valid_status(self, normalized_data: Dict[str, Any]) -> None:
        """
        Ensure the status is valid.
        
        Args:
            normalized_data: The normalized data dictionary
        """
        valid_statuses = ["active", "closed", "awarded", "canceled", "upcoming", "unknown"]
        
        # Handle incorrect tender type in status field
        if "status" in normalized_data and normalized_data["status"]:
            if isinstance(normalized_data["status"], str):
                status_lower = normalized_data["status"].lower()
                
                # Handle common procurement methods that should not be in status
                procurement_methods = [
                    "request for proposal", "request for quotation", "invitation to bid", 
                    "call for individual consultants", "request for expression of interest",
                    "expression of interest", "rfi", "rfp", "rfq", "itb"
                ]
                
                if status_lower in procurement_methods:
                    # Move this value to tender_type if it's a procurement method
                    if status_lower not in valid_statuses:
                        # Save this as procurement_method if no value exists
                        if "procurement_method" not in normalized_data or not normalized_data["procurement_method"]:
                            normalized_data["procurement_method"] = normalized_data["status"]
                        
                        # Also use this value for tender_type if none exists
                        if "tender_type" not in normalized_data or not normalized_data["tender_type"]:
                            normalized_data["tender_type"] = normalized_data["status"]
                        
                        # Reset status to a valid value (will be set by date inference)
                        normalized_data["status"] = "unknown"
                
                # If still invalid, set to unknown
                if normalized_data["status"].lower() not in valid_statuses:
                    normalized_data["status"] = "unknown"
        
        # Always re-apply date-based status inference as a last resort
        if "status" not in normalized_data or normalized_data["status"] == "unknown":
            self._infer_status_from_dates(normalized_data)

    async def normalize_tender(self, tender: RawTender, save_debug: bool = False) -> Dict[str, Any]:
        """
        Normalize a single tender.
        
        Args:
            tender: The raw tender data
            save_debug: Whether to save debug data
            
        Returns:
            The normalized tender data
        """
        start_time = time.time()
        tender_id = tender.id
        source = tender.source_table
        
        self.logger.info(f"Normalizing tender {tender_id} from {source}")
        
        should_use_llm, reason = self._should_use_llm(tender)
        
        if should_use_llm:
            self.logger.info(f"Using LLM for tender {tender_id}: {reason}")
            
            # Prepare the normalization input
            input_data = NormalizationInput(
                id=tender.id,
                source_table=tender.source_table,
                title=tender.title,
                description=tender.description,
                publication_date=tender.publication_date,
                deadline_date=tender.deadline_date,
                country=tender.country,
                organization_name=tender.organization_name,
                source_data=tender.source_data,
                raw_tender=tender.model_dump()
            )
            
            try:
                # Try PydanticAI first
                self.logger.info(f"Attempting to normalize tender {tender_id} with PydanticAI")
                try:
                    # Call the LLM normalization function
                    output = await self._normalize_with_llm(input_data)
                    
                    # Update metrics
                    self._record_llm_usage(True)
                    end_time = time.time()
                    processing_time = end_time - start_time
                    
                    return {
                        "normalized_data": output.normalized_data,
                        "used_llm": True,
                        "method": "pydantic_ai",
                        "processing_time": processing_time,
                        "error": output.error if hasattr(output, "error") else None,
                        "missing_fields": output.missing_fields if hasattr(output, "missing_fields") else [],
                        "notes": output.notes if hasattr(output, "notes") else None
                    }
                except Exception as e:
                    error_str = str(e)
                    self.logger.error(f"PydanticAI normalization failed: {error_str}")
                    
                    # If the error is related to the "Expected code to be unreachable" issue
                    # or any other PydanticAI serialization problem, try DirectNormalizer
                    if "Expected code to be unreachable" in error_str or "serialization" in error_str.lower():
                        self.logger.info(f"Falling back to DirectNormalizer for tender {tender_id}")
                        
                        # Import here to avoid circular imports
                        from .direct_normalizer import DirectNormalizer
                        
                        # Initialize DirectNormalizer with API key from settings
                        direct_normalizer = DirectNormalizer(
                            api_key=settings.openai_api_key.get_secret_value(),
                            model=settings.openai_model
                        )
                        
                        try:
                            # Call DirectNormalizer
                            direct_result = await direct_normalizer.normalize_tender(
                                tender.model_dump(), 
                                save_debug=save_debug
                            )
                            
                            # Update metrics
                            self._record_llm_usage(True)
                            end_time = time.time()
                            processing_time = end_time - start_time
                            
                            return {
                                "normalized_data": direct_result.get("normalized_data", {}),
                                "used_llm": True,
                                "method": "direct_normalizer",
                                "processing_time": processing_time,
                                "error": None,
                                "missing_fields": [],
                                "notes": "Used DirectNormalizer as fallback due to PydanticAI serialization issues"
                            }
                        except Exception as direct_error:
                            self.logger.error(f"DirectNormalizer fallback also failed: {str(direct_error)}")
                            self.logger.error(traceback.format_exc())
                            
                            # Try MockNormalizer as a final fallback
                            self.logger.info(f"Falling back to MockNormalizer for tender {tender_id}")
                            
                            # Import here to avoid circular imports
                            from .mock_normalizer import MockNormalizer
                            
                            # Initialize MockNormalizer
                            mock_normalizer = MockNormalizer(
                                api_key=settings.openai_api_key.get_secret_value(),
                                model=settings.openai_model
                            )
                            
                            try:
                                # Call MockNormalizer
                                mock_result = await mock_normalizer.normalize_tender(tender.model_dump())
                                
                                # Update metrics
                                self._record_llm_usage(False)  # Not using LLM for mock
                                end_time = time.time()
                                processing_time = end_time - start_time
                                
                                return {
                                    "normalized_data": mock_result.get("normalized_data", {}),
                                    "used_llm": False,
                                    "method": "mock_normalizer",
                                    "processing_time": processing_time,
                                    "error": None,
                                    "missing_fields": mock_result.get("missing_fields", []),
                                    "notes": "Used MockNormalizer as fallback due to API errors"
                                }
                            except Exception as mock_error:
                                self.logger.error(f"MockNormalizer fallback also failed: {str(mock_error)}")
                                self.logger.error(traceback.format_exc())
                                # Continue to fallback method
                    else:
                        # Re-raise if not a serialization error
                        raise
            except Exception as e:
                self.logger.error(f"LLM normalization failed: {str(e)}")
                self.logger.error(traceback.format_exc())
                
                # Record failure but continue with fallback
                self._record_llm_usage(False)
                
                # Try MockNormalizer as a final fallback for any LLM failure
                self.logger.info(f"Falling back to MockNormalizer for tender {tender_id} after LLM failure")
                
                # Import here to avoid circular imports
                from .mock_normalizer import MockNormalizer
                
                # Initialize MockNormalizer
                mock_normalizer = MockNormalizer(
                    api_key=settings.openai_api_key.get_secret_value(),
                    model=settings.openai_model
                )
                
                try:
                    # Call MockNormalizer
                    mock_result = await mock_normalizer.normalize_tender(tender.model_dump())
                    
                    # Update metrics
                    end_time = time.time()
                    processing_time = end_time - start_time
                    
                    return {
                        "normalized_data": mock_result.get("normalized_data", {}),
                        "used_llm": False,
                        "method": "mock_normalizer",
                        "processing_time": processing_time,
                        "error": None,
                        "missing_fields": mock_result.get("missing_fields", []),
                        "notes": "Used MockNormalizer as fallback due to LLM failure"
                    }
                except Exception as mock_error:
                    self.logger.error(f"MockNormalizer fallback also failed: {str(mock_error)}")
                    self.logger.error(traceback.format_exc())
                    # Continue to fallback method
        else:
            self.logger.info(f"Using direct parsing for tender {tender_id}: {reason}")
        
        # Fallback to direct parsing or if LLM was disabled
        self.logger.info(f"Using direct parsing for tender {tender_id}")
        
        # Create empty normalized data with original fields
        normalized_data = {
            "id": tender.id,
            "source_table": tender.source_table,
            "title": tender.title,
            "description": tender.description,
            "country": tender.country,
            "organization_name": tender.organization_name,
        }
        
        # Add dates if available
        if tender.publication_date:
            normalized_data["publication_date"] = tender.publication_date
        if tender.deadline_date:
            normalized_data["deadline_date"] = tender.deadline_date
        
        # Record performance metrics
        end_time = time.time()
        processing_time = end_time - start_time
        
        return {
            "normalized_data": normalized_data,
            "used_llm": False,
            "method": "direct_parsing",
            "processing_time": processing_time,
            "error": "Fallback to direct parsing" if should_use_llm else "Direct parsing used as configured",
            "missing_fields": [],
            "notes": "Used fallback method" if should_use_llm else None
        }

    def normalize_tender_sync(self, tender: RawTender, save_debug: bool = False) -> Dict[str, Any]:
        """
        Synchronous wrapper for normalize_tender.
        """
        import asyncio
        
        # Create an event loop or get the current one
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
        # Run the async function
        return loop.run_until_complete(self.normalize_tender(tender, save_debug))

    def _record_llm_usage(self, success: bool) -> None:
        """
        Record LLM usage statistics.
        
        Args:
            success: Whether the LLM call was successful
        """
        self.performance_stats["total_processed"] += 1
        self.performance_stats["llm_used"] += 1
        
        if success:
            self.performance_stats["success_rate"] = (
                self.performance_stats["llm_used"] / self.performance_stats["total_processed"]
                if self.performance_stats["total_processed"] > 0
                else 0
            )
        else:
            self.performance_stats["fallback_used"] += 1


# Create a singleton instance
normalizer = TenderNormalizer() 