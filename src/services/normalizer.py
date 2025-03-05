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
    error: Optional[str] = Field(
        None, description="Error message if normalization failed"
    )
    
    # Field validation
    @model_validator(mode="after")
    def validate_tender_fields(self) -> "NormalizationOutput":
        """Validate that the tender has required fields and correct types."""
        # Skip validation if there was an error
        if self.error:
            return self
            
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
    """
    Service for normalizing tender data using pure Pydantic parsing.
    """
    
    def __init__(self):
        """Initialize the TenderNormalizer."""
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing TenderNormalizer")
        
        # Add config attribute
        self.config = NormalizerConfig()
            
        # Performance tracking
        self.performance_stats = {
            "total_processed": 0,
            "normalization_time": 0,
            "success_rate": 0,
        }
    
    def _save_debug_data(self, data: Any, data_type: str) -> None:
        """
        Save debug data to a file.
        
        Args:
            data: The data to save
            data_type: The type of data (input, output, error, etc.)
        """
        if not self.config.save_debug_data:
            return
            
        # Create debug directory if it doesn't exist
        debug_dir = Path("debug_dumps")
        debug_dir.mkdir(exist_ok=True)
        
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"{data_type}_{timestamp}.json"
        
        # If it's input data, add source and ID
        if isinstance(data, NormalizationInput) and data_type == "input":
            source = data.source_table
            tender_id = data.id
            filename = f"input_{source}_{tender_id}_{timestamp}.json"
        
        # Save data to file
        file_path = debug_dir / filename
        
        try:
            # Convert data to JSON
            if hasattr(data, "model_dump"):
                json_data = data.model_dump()
            elif isinstance(data, (dict, list)):
                json_data = data
            else:
                json_data = {"data": str(data)}
                
            # Write to file
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False, default=str)
                
            self.logger.info(f"Saved debug data to {file_path}")
        except Exception as e:
            self.logger.error(f"Failed to save debug data: {str(e)}")
    
    def _create_normalization_input(self, tender: RawTender) -> NormalizationInput:
        """
        Create a NormalizationInput object from a RawTender.
        
        Args:
            tender: The raw tender data
            
        Returns:
            NormalizationInput object
        """
        return NormalizationInput(
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
        
    async def normalize_tender(self, tender: RawTender, save_debug: bool = True) -> Dict[str, Any]:
        """
        Normalize a tender using direct parsing.
        
        Args:
            tender: The tender to normalize
            save_debug: Whether to save debug data
            
        Returns:
            Dictionary with normalized data and metadata
        """
        tender_id = tender.id
        start_time = time.time()
        
        self.logger.info(f"Normalizing tender {tender_id} from {tender.source_table}")
        
        # Create input data for normalization
        input_data = self._create_normalization_input(tender)
        
        # Save debug data if enabled
        if save_debug and self.config.save_debug_data:
            self._save_debug_data(input_data, f"input_{tender.source_table}_{tender_id}")
        
        try:
            # Use direct parsing approach for all tenders
            self.logger.info(f"Using direct parsing for tender {tender_id}")
            
            # Create normalized data with basic fields
            normalized_data = {
                "id": tender.id,
                "source_table": tender.source_table,
                "source_id": tender.id,
                "title": format_title(tender.title or ""),
                "description": tender.description or "",
                "country": tender.country or "",
                "organization_name": tender.organization_name or "",
            }
            
            # Detect and translate title if not in English
            if tender.title:
                title_lang = detect_language(tender.title)
                if title_lang != "en" and title_lang != "unknown":
                    try:
                        normalized_data["title"] = format_title(translate_to_english(tender.title, title_lang))
                        normalized_data["original_language"] = title_lang
                    except Exception as e:
                        self.logger.warning(f"Failed to translate title: {str(e)}")
            
            # Detect and translate description if not in English
            if tender.description and len(tender.description) > 10:
                desc_lang = detect_language(tender.description[:1000])  # Only check first 1000 chars for efficiency
                if desc_lang != "en" and desc_lang != "unknown":
                    try:
                        normalized_data["description"] = translate_to_english(tender.description, desc_lang)
                        normalized_data["original_language"] = desc_lang
                    except Exception as e:
                        self.logger.warning(f"Failed to translate description: {str(e)}")
            
            # Extract URLs from description
            if tender.description:
                urls = extract_urls_from_text(tender.description)
                if urls:
                    normalized_data["extracted_urls"] = urls
            
            # Extract emails from description
            if tender.description:
                emails = extract_emails_from_text(tender.description)
                if emails:
                    normalized_data["extracted_emails"] = emails
            
            # Add dates if available
            if hasattr(tender, "publication_date") and tender.publication_date:
                normalized_data["publication_date"] = tender.publication_date
            if hasattr(tender, "deadline_date") and tender.deadline_date:
                normalized_data["deadline_date"] = tender.deadline_date
            
            # Extract URL
            if hasattr(tender, "url") and tender.url:
                normalized_data["url"] = tender.url
            
            # Try to extract additional fields from source_data if available
            if hasattr(tender, "source_data") and tender.source_data:
                source_data = tender.source_data
                
                # Recursively extract all fields from source_data
                self._extract_fields_from_source_data(source_data, normalized_data)
                
                # Extract URL if available
                if "url" in source_data:
                    normalized_data["url"] = source_data["url"]
                elif "link" in source_data:
                    normalized_data["url"] = source_data["link"]
                
                # Extract tender type if available
                if "tender_type" in source_data:
                    normalized_data["tender_type"] = source_data["tender_type"]
                elif "type" in source_data:
                    normalized_data["tender_type"] = source_data["type"]
                elif "procurement_type" in source_data:
                    normalized_data["tender_type"] = source_data["procurement_type"]
                else:
                    normalized_data["tender_type"] = "unknown"
                
                # Extract status if available
                if "status" in source_data:
                    normalized_data["status"] = source_data["status"]
                else:
                    # Infer status from dates
                    normalized_data["status"] = self._infer_status_from_dates(normalized_data)
                
                # Extract organization details if available
                if "organization" in source_data:
                    org_data = source_data["organization"]
                    if isinstance(org_data, dict):
                        if "name" in org_data and not normalized_data.get("organization_name"):
                            normalized_data["organization_name"] = org_data["name"]
                        if "id" in org_data:
                            normalized_data["organization_id"] = org_data["id"]
                
                # Extract specific fields for different source tables
                self._extract_source_specific_fields(tender.source_table, source_data, normalized_data)
            
            # Ensure status is set
            if "status" not in normalized_data:
                normalized_data["status"] = self._infer_status_from_dates(normalized_data)
            
            # Update metrics
            end_time = time.time()
            processing_time = end_time - start_time
            
            return {
                "normalized_data": normalized_data,
                "used_llm": False,
                "method": "direct_parsing",
                "processing_time": processing_time,
                "error": None,
                "missing_fields": [],
                "notes": "Normalized using direct parsing approach"
            }
            
        except Exception as e:
            self.logger.error(f"Error in normalize_tender: {str(e)}")
            self.logger.error(traceback.format_exc())
            
            # Return error output
            end_time = time.time()
            processing_time = end_time - start_time
            
            return {
                "normalized_data": {
                    "id": tender.id,
                    "source_table": tender.source_table,
                    "source_id": tender.id,
                    "title": tender.title or "",
                    "description": "Error during normalization",
                },
                "used_llm": False,
                "method": "direct_parsing_error",
                "processing_time": processing_time,
                "error": str(e),
                "missing_fields": [],
                "notes": f"Normalization failed: {str(e)}"
            }
    
    def _extract_fields_from_source_data(self, source_data: Dict[str, Any], normalized_data: Dict[str, Any]) -> None:
        """
        Recursively extract all fields from source_data.
        
        Args:
            source_data: The source data dictionary
            normalized_data: The normalized data dictionary to update
        """
        # Extract direct fields
        for field in [
            "buyer", "project_name", "project_id", "project_number", 
            "sector", "estimated_value", "currency", "contact_name", 
            "contact_email", "contact_phone", "contact_address", 
            "document_links", "language", "notice_id", "reference_number", 
            "procurement_method", "funding_source", "location", "city",
            "region", "postal_code", "address", "classification",
            "value", "procurement_category", "award_criteria"
        ]:
            if field in source_data:
                normalized_data[field] = source_data[field]
        
        # Look for nested fields with different names
        for field, nested_fields in {
            "details": ["value", "category", "method", "criteria", "type", "status"],
            "dates": ["published", "updated", "closing", "deadline", "award", "start", "end"],
            "contact": ["name", "email", "phone", "address"],
            "location": ["country", "city", "region", "address"],
            "organization": ["name", "id", "type", "address"],
            "project": ["name", "id", "number", "description"],
            "value": ["amount", "currency"],
            "documents": ["urls", "links", "attachments"]
        }.items():
            if field in source_data and isinstance(source_data[field], dict):
                for subfield in nested_fields:
                    if subfield in source_data[field]:
                        normalized_data[f"{field}_{subfield}"] = source_data[field][subfield]
    
    def _extract_source_specific_fields(self, source_table: str, source_data: Dict[str, Any], normalized_data: Dict[str, Any]) -> None:
        """
        Extract fields specific to different source tables.
        
        Args:
            source_table: The source table name
            source_data: The source data dictionary
            normalized_data: The normalized data dictionary to update
        """
        # Sam.gov specific fields
        if source_table == "sam_gov":
            for field in ["notice_id", "solnbr", "agency", "office", "location", "zip", "naics"]:
                if field in source_data:
                    normalized_data[field] = source_data[field]
        
        # World Bank specific fields
        elif source_table == "wb":
            for field in ["borrower", "project_id", "loan_number", "selection_method"]:
                if field in source_data:
                    normalized_data[field] = source_data[field]
        
        # Asian Development Bank specific fields
        elif source_table == "adb":
            for field in ["project_number", "sector", "loan_number", "closing_date"]:
                if field in source_data:
                    normalized_data[field] = source_data[field]
        
        # TED EU specific fields
        elif source_table == "ted_eu":
            for field in ["document_number", "regulation", "notice_type", "award_criteria"]:
                if field in source_data:
                    normalized_data[field] = source_data[field]
        
        # UN Global Marketplace specific fields
        elif source_table == "ungm":
            for field in ["reference", "deadline_timezone", "vendor_city", "vendor_country"]:
                if field in source_data:
                    normalized_data[field] = source_data[field]
        
        # AFD Tenders specific fields
        elif source_table == "afd_tenders":
            for field in ["country_code", "project_ref", "submission_method", "language_code"]:
                if field in source_data:
                    normalized_data[field] = source_data[field]
        
        # Inter-American Development Bank specific fields
        elif source_table == "iadb":
            for field in ["operation_number", "operation_type", "financing_type", "sector_code"]:
                if field in source_data:
                    normalized_data[field] = source_data[field]
        
        # African Development Bank specific fields
        elif source_table == "afdb":
            for field in ["country_code", "funding_source", "sector_code", "task_manager"]:
                if field in source_data:
                    normalized_data[field] = source_data[field]
        
        # Asian Infrastructure Investment Bank specific fields
        elif source_table == "aiib":
            for field in ["borrower", "sector", "project_status", "project_timeline"]:
                if field in source_data:
                    normalized_data[field] = source_data[field]
    
    def _infer_status_from_dates(self, data: Dict[str, Any]) -> str:
        """
        Infer the tender status from publication and deadline dates.
        
        Args:
            data: The normalized data with dates
            
        Returns:
            The inferred status
        """
        today = datetime.now().date()
        
        # Parse dates if they exist
        publication_date = None
        deadline_date = None
        
        if "publication_date" in data and data["publication_date"]:
            try:
                if isinstance(data["publication_date"], str):
                    publication_date = datetime.fromisoformat(data["publication_date"]).date()
                elif isinstance(data["publication_date"], datetime):
                    publication_date = data["publication_date"].date()
            except (ValueError, TypeError):
                pass
                
        if "deadline_date" in data and data["deadline_date"]:
            try:
                if isinstance(data["deadline_date"], str):
                    deadline_date = datetime.fromisoformat(data["deadline_date"]).date()
                elif isinstance(data["deadline_date"], datetime):
                    deadline_date = data["deadline_date"].date()
            except (ValueError, TypeError):
                pass
        
        # Infer status based on dates
        if deadline_date:
            if deadline_date < today:
                return "closed"
            else:
                return "active"
        elif publication_date:
            if publication_date > today:
                return "upcoming"
            else:
                return "active"
        
        # Default status if no dates are available
        return "unknown"
    
    def normalize_tender_sync(self, tender: RawTender, save_debug: bool = True) -> Dict[str, Any]:
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
    
    async def normalize_test_batch(self, tenders_by_source: Dict[str, List[RawTender]]) -> Dict[str, List[NormalizationResult]]:
        """
        Normalize a batch of tenders grouped by source for testing purposes.
        
        Args:
            tenders_by_source: Dictionary mapping source names to lists of RawTender objects
            
        Returns:
            Dictionary mapping source names to lists of NormalizationResult objects
        """
        results = {}
        
        for source, tenders in tenders_by_source.items():
            self.logger.info(f"Processing {len(tenders)} test tenders from {source}")
            source_results = []
            
            for tender in tenders:
                start_time = time.time()
                fields_before = len([f for f in tender.model_dump() if tender.model_dump().get(f)])
                
                try:
                    # Normalize the tender
                    normalized_data = await self.normalize_tender(tender, save_debug=True)
                    
                    # Calculate processing time
                    processing_time = time.time() - start_time
                    
                    # Check if normalization was successful
                    if normalized_data and "normalized_data" in normalized_data:
                        # Create normalized tender object
                        normalized_tender = NormalizedTender(
                            **normalized_data["normalized_data"],
                            normalized_method=normalized_data.get("method", "unknown"),
                            processing_time_ms=int(processing_time * 1000)
                        )
                        
                        # Count fields after normalization
                        fields_after = len([f for f in normalized_tender.model_dump() if normalized_tender.model_dump().get(f)])
                        
                        # Calculate improvement percentage
                        improvement = ((fields_after - fields_before) / fields_before) * 100 if fields_before > 0 else 0
                        
                        # Create successful result
                        result = NormalizationResult(
                            tender_id=tender.id,
                            source_table=tender.source_table,
                            success=True,
                            normalized_tender=normalized_tender,
                            error=None,
                            processing_time=processing_time,
                            method_used=normalized_data.get("method", "unknown"),
                            fields_before=fields_before,
                            fields_after=fields_after,
                            improvement_percentage=improvement
                        )
                    else:
                        # Create failed result
                        result = NormalizationResult(
                            tender_id=tender.id,
                            source_table=tender.source_table,
                            success=False,
                            normalized_tender=None,
                            error="Normalization returned empty or invalid data",
                            processing_time=processing_time,
                            method_used="failed",
                            fields_before=fields_before,
                            fields_after=0,
                            improvement_percentage=0
                        )
                except Exception as e:
                    # Calculate processing time
                    processing_time = time.time() - start_time
                    
                    # Log the error
                    self.logger.error(f"Error normalizing test tender {tender.id}: {str(e)}")
                    
                    # Create error result
                    result = NormalizationResult(
                        tender_id=tender.id,
                        source_table=tender.source_table,
                        success=False,
                        normalized_tender=None,
                        error=str(e),
                        processing_time=processing_time,
                        method_used="failed",
                        fields_before=fields_before,
                        fields_after=0,
                        improvement_percentage=0
                    )
                
                # Add result to source results
                source_results.append(result)
                
                # Log result
                status = "SUCCESS" if result.success else "FAILED"
                self.logger.info(f"Normalized test tender {tender.id}: {status} (method: {result.method_used}, time: {result.processing_time:.2f}s)")
                
            # Add source results to results
            results[source] = source_results
            
            # Log source summary
            successful = len([r for r in source_results if r.success])
            self.logger.info(f"Completed {source} test batch: {successful}/{len(source_results)} successful")
            
        return results


# Create a singleton instance
normalizer = TenderNormalizer() 