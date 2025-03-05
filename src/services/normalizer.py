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
from datetime import datetime, date, timedelta
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
            "success_count": 0,
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
            
            # Get normalized data with source-specific defaults
            normalized_data = self._parse_fields_directly(tender)
            
            # Update metrics
            end_time = time.time()
            processing_time = end_time - start_time
            
            # Track performance
            self.performance_stats["total_processed"] += 1
            self.performance_stats["normalization_time"] += processing_time
            
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
            # Log the error
            error_message = str(e)
            self.logger.error(f"Error normalizing tender {tender_id}: {error_message}")
            
            # Update metrics
            end_time = time.time()
            processing_time = end_time - start_time
            
            return {
                "normalized_data": None,
                "used_llm": False,
                "method": "direct_parsing_error",
                "processing_time": processing_time,
                "error": error_message,
                "missing_fields": [],
                "notes": f"Error during direct parsing: {error_message}"
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
        # Default country mappings for sources that don't explicitly provide country
        source_country_defaults = {
            "sam_gov": "United States",
            "ted_eu": "European Union",
            "ungm": "International",
            "afd_tenders": "France",  # Default for French Development Agency
            "iadb": "International",  # Inter-American Development Bank
            "afdb": "International",  # African Development Bank
            "aiib": "International",  # Asian Infrastructure Investment Bank
            "adb": "International"    # Asian Development Bank
        }
        
        # Apply default country if not set and available in defaults
        if (not normalized_data.get("country") or normalized_data.get("country") == "") and source_table in source_country_defaults:
            normalized_data["country"] = source_country_defaults[source_table]
        
        # Sam.gov specific fields
        if source_table == "sam_gov":
            for field in ["notice_id", "solnbr", "agency", "office", "location", "zip", "naics"]:
                if field in source_data:
                    normalized_data[field] = source_data[field]
            
            # ALWAYS set United States as the country for all sam.gov tenders
            normalized_data["country"] = "United States"
            
            # Try to extract organization name from agency field if not already set
            if (not normalized_data.get("organization_name") or normalized_data.get("organization_name") == "") and "agency" in source_data:
                normalized_data["organization_name"] = source_data["agency"]
        
        # World Bank specific fields
        elif source_table == "wb":
            for field in ["borrower", "project_id", "loan_number", "selection_method"]:
                if field in source_data:
                    normalized_data[field] = source_data[field]
            
            # Extract country from borrower or location fields if available
            if "borrower" in source_data and source_data.get("borrower") and (not normalized_data.get("country") or normalized_data.get("country") == ""):
                normalized_data["country"] = source_data["borrower"]
            
            if "location" in source_data and source_data.get("location") and (not normalized_data.get("country") or normalized_data.get("country") == ""):
                normalized_data["country"] = source_data["location"]
            
            # Default to "International" if no country found
            if not normalized_data.get("country") or normalized_data.get("country") == "":
                normalized_data["country"] = "International"
        
        # Asian Development Bank specific fields
        elif source_table == "adb":
            for field in ["project_number", "sector", "loan_number", "closing_date"]:
                if field in source_data:
                    normalized_data[field] = source_data[field]
            
            # Ensure title is set - critical for ADB tenders based on logs
            if not normalized_data.get("title") or normalized_data.get("title") == "":
                if source_data.get("project_name"):
                    normalized_data["title"] = source_data["project_name"]
                elif source_data.get("name"):
                    normalized_data["title"] = source_data["name"]
                elif source_data.get("project_title"):
                    normalized_data["title"] = source_data["project_title"]
                elif source_data.get("subject"):
                    normalized_data["title"] = source_data["subject"]
                else:
                    # Create a title from project number and country if available
                    project_number = source_data.get("project_number") or normalized_data.get("id") or "Unknown"
                    country = normalized_data.get("country") or "Unknown Location"
                    normalized_data["title"] = f"ADB Project {project_number} - {country}"
            
            # Ensure country is set for ADB
            if not normalized_data.get("country") or normalized_data.get("country") == "":
                if source_data.get("country"):
                    normalized_data["country"] = source_data["country"]
                else:
                    normalized_data["country"] = "International"
        
        # TED EU specific fields
        elif source_table == "ted_eu":
            for field in ["document_number", "regulation", "notice_type", "award_criteria"]:
                if field in source_data:
                    normalized_data[field] = source_data[field]
            
            # ALWAYS set EU as country - critical based on logs
            normalized_data["country"] = "European Union"
            
            # Try to extract specific EU country if available
            if source_data.get("country"):
                normalized_data["country"] = source_data["country"]
            elif source_data.get("member_state"):
                normalized_data["country"] = source_data["member_state"]
        
        # UN Global Marketplace specific fields
        elif source_table == "ungm":
            for field in ["reference", "deadline_timezone", "vendor_city", "vendor_country"]:
                if field in source_data:
                    normalized_data[field] = source_data[field]
            
            # ALWAYS set a country value for UNGM - critical based on logs
            if source_data.get("vendor_country"):
                normalized_data["country"] = source_data["vendor_country"]
            else:
                normalized_data["country"] = "International"  # Default for UNGM
        
        # AFD Tenders specific fields
        elif source_table == "afd_tenders":
            for field in ["country_code", "project_ref", "submission_method", "language_code"]:
                if field in source_data:
                    normalized_data[field] = source_data[field]
            
            # Ensure country is set
            if not normalized_data.get("country") or normalized_data.get("country") == "":
                if source_data.get("country"):
                    normalized_data["country"] = source_data["country"]
                else:
                    normalized_data["country"] = "France"  # Default for AFD
            
            # Fix date parsing for AFD tenders - critical issue based on logs
            if "publication_date" in normalized_data and isinstance(normalized_data["publication_date"], str):
                try:
                    # Handle various date formats
                    date_str = normalized_data["publication_date"]
                    if "," in date_str:  # Format like "Feb 6, 2025"
                        from dateutil import parser
                        parsed_date = parser.parse(date_str).strftime("%Y-%m-%d")
                        normalized_data["publication_date"] = parsed_date
                except Exception as e:
                    self.logger.warning(f"Failed to parse publication_date for AFD tender: {e}")
                    # Use current date as fallback
                    normalized_data["publication_date"] = datetime.now().strftime("%Y-%m-%d")
            
            if "deadline_date" in normalized_data and isinstance(normalized_data["deadline_date"], str):
                try:
                    # Handle various date formats
                    date_str = normalized_data["deadline_date"]
                    if "," in date_str:  # Format like "Feb 6, 2025"
                        from dateutil import parser
                        parsed_date = parser.parse(date_str).strftime("%Y-%m-%d")
                        normalized_data["deadline_date"] = parsed_date
                except Exception as e:
                    self.logger.warning(f"Failed to parse deadline_date for AFD tender: {e}")
                    # Use a future date as fallback for deadline
                    normalized_data["deadline_date"] = (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d")
        
        # Inter-American Development Bank specific fields
        elif source_table == "iadb":
            for field in ["operation_number", "operation_type", "financing_type", "sector_code"]:
                if field in source_data:
                    normalized_data[field] = source_data[field]
            
            # Ensure title is set for IADB - critical based on logs
            if not normalized_data.get("title") or normalized_data.get("title") == "":
                if source_data.get("project_name"):
                    normalized_data["title"] = source_data["project_name"]
                elif source_data.get("operation_name"):
                    normalized_data["title"] = source_data["operation_name"]
                elif source_data.get("name"):
                    normalized_data["title"] = source_data["name"]
                else:
                    # Create a title from operation number if available
                    operation_num = source_data.get("operation_number") or normalized_data.get("id") or "Unknown"
                    country = normalized_data.get("country") or "International"
                    normalized_data["title"] = f"IADB Project {operation_num} - {country}"
            
            # Ensure country is set
            if not normalized_data.get("country") or normalized_data.get("country") == "":
                if source_data.get("country"):
                    normalized_data["country"] = source_data["country"]
                elif source_data.get("operation_country"):
                    normalized_data["country"] = source_data["operation_country"]
                else:
                    normalized_data["country"] = "International"
        
        # African Development Bank specific fields
        elif source_table == "afdb":
            for field in ["country_code", "funding_source", "sector_code", "task_manager"]:
                if field in source_data:
                    normalized_data[field] = source_data[field]
            
            # Extract country if available
            if source_data.get("country"):
                normalized_data["country"] = source_data["country"]
            elif source_data.get("country_name"):
                normalized_data["country"] = source_data["country_name"]
            else:
                normalized_data["country"] = "International"
            
            # Fix date parsing for AFDB tenders - critical based on logs
            if "publication_date" in normalized_data and normalized_data.get("publication_date") in ["Unknown", "unknown", ""]:
                # Use current date as fallback for publication date
                normalized_data["publication_date"] = datetime.now().strftime("%Y-%m-%d")
        
        # Asian Infrastructure Investment Bank specific fields
        elif source_table == "aiib":
            for field in ["borrower", "sector", "project_status", "project_timeline"]:
                if field in source_data:
                    normalized_data[field] = source_data[field]
            
            # ALWAYS set a country value for AIIB - critical based on logs
            if source_data.get("borrower"):
                normalized_data["country"] = source_data["borrower"]
            else:
                normalized_data["country"] = "International"  # Default for AIIB
        
        # Handle any missing required fields for all sources
        if not normalized_data.get("country") or normalized_data.get("country") == "":
            normalized_data["country"] = "International"  # Default fallback
        
        if not normalized_data.get("title") or normalized_data.get("title") == "":
            source_id = normalized_data.get("id") or "Unknown"
            normalized_data["title"] = f"Tender {source_id} from {source_table}"
        
        if not normalized_data.get("description") or normalized_data.get("description") == "":
            title = normalized_data.get("title") or "Unknown Tender"
            normalized_data["description"] = f"Details for {title}"
            
        # Ensure dates are properly formatted
        for date_field in ["publication_date", "deadline_date"]:
            if date_field in normalized_data and isinstance(normalized_data[date_field], str):
                # Try to parse and standardize date format
                try:
                    if normalized_data[date_field] in ["Unknown", "unknown", ""]:
                        if date_field == "publication_date":
                            normalized_data[date_field] = datetime.now().strftime("%Y-%m-%d")
                        else:  # deadline_date
                            normalized_data[date_field] = (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d")
                    elif "," in normalized_data[date_field]:  # Format like "Feb 6, 2025"
                        from dateutil import parser
                        parsed_date = parser.parse(normalized_data[date_field]).strftime("%Y-%m-%d")
                        normalized_data[date_field] = parsed_date
                except Exception as e:
                    self.logger.warning(f"Failed to parse {date_field}: {e}")
                    if date_field == "publication_date":
                        normalized_data[date_field] = datetime.now().strftime("%Y-%m-%d")
                    else:  # deadline_date
                        normalized_data[date_field] = (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d")
    
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
    
    def normalize_tender_sync(self, tender: RawTender, source_table: Optional[str] = None) -> Dict[str, Any]:
        """
        Synchronously normalize a tender.
        
        Args:
            tender: The tender to normalize
            source_table: Optional source table name override
            
        Returns:
            Dictionary with normalized data and metadata
        """
        try:
            if source_table:
                # Override source table if provided
                tender.source_table = source_table
            
            # Create a new event loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(self.normalize_tender(tender))
            loop.close()
            
            # Update success metrics if normalization was successful
            if result.get("normalized_data") is not None:
                # Increment success count
                self.performance_stats["success_count"] += 1
                
                # Calculate success rate
                total = self.performance_stats.get("total_processed", 1)
                self.performance_stats["success_rate"] = (self.performance_stats["success_count"] / total) * 100
            
            return result
            
        except Exception as e:
            # Return error output
            self.logger.error(f"Error in normalize_tender_sync: {str(e)}")
            
            return {
                "normalized_data": None,
                "used_llm": False,
                "method": "failed",
                "processing_time": 0,
                "error": str(e),
                "missing_fields": [],
                "notes": f"Error during synchronous normalization: {str(e)}"
            }
    
    async def normalize_test_batch(self, tenders_by_source: Dict[str, List[RawTender]]) -> Dict[str, List[NormalizationResult]]:
        """
        Normalize a batch of test tenders from multiple sources.
        
        Args:
            tenders_by_source: Dictionary mapping source table names to lists of RawTender objects
            
        Returns:
            Dictionary mapping source table names to lists of NormalizationResult objects
        """
        results_by_source = {}
        
        # Process each source table
        for source_table, tenders in tenders_by_source.items():
            self.logger.info(f"Processing {len(tenders)} test tenders from {source_table}")
            results = []
            
            success_count = 0
            
            # Process each tender in the source table
            for tender in tenders:
                start_time = time.time()
                
                try:
                    # Normalize the tender
                    self.logger.info(f"Normalizing tender {tender.id} from {source_table}")
                    
                    # Save debug data if enabled
                    if self.config.save_debug_data:
                        self._save_debug_data(self._create_normalization_input(tender), f"input_{source_table}_{tender.id}")
                    
                    # Call the normalize_tender method
                    result = await self.normalize_tender(tender)
                    
                    # Check if normalization was successful
                    if result.get("normalized_data") is not None:
                        success_count += 1
                        method = result.get("method", "unknown")
                    else:
                        method = "failed"
                    
                    # Calculate processing time
                    end_time = time.time()
                    processing_time = end_time - start_time
                    
                    # Log the result
                    self.logger.info(f"Normalized test tender {tender.id}: {'SUCCESS' if result.get('normalized_data') is not None else 'FAILED'} (method: {method}, time: {processing_time:.2f}s)")
                    
                    # Add to results
                    results.append({
                        "tender_id": tender.id,
                        "source_table": source_table,
                        "success": result.get("normalized_data") is not None,
                        "method": method,
                        "processing_time": processing_time,
                        "normalized_data": result.get("normalized_data"),
                        "error": result.get("error")
                    })
                    
                except Exception as e:
                    # Log the error
                    self.logger.error(f"Error normalizing test tender {tender.id}: {str(e)}")
                    
                    # Calculate processing time
                    end_time = time.time()
                    processing_time = end_time - start_time
                    
                    # Log the result
                    self.logger.info(f"Normalized test tender {tender.id}: FAILED (method: failed, time: {processing_time:.2f}s)")
                    
                    # Add to results
                    results.append({
                        "tender_id": tender.id,
                        "source_table": source_table,
                        "success": False,
                        "method": "failed",
                        "processing_time": processing_time,
                        "normalized_data": None,
                        "error": str(e)
                    })
            
            # Log the success rate for this source table
            self.logger.info(f"Completed {source_table} test batch: {success_count}/{len(tenders)} successful")
            
            # Add to results by source
            results_by_source[source_table] = results
        
        return results_by_source

    def normalize_test_tender(self, tender: dict, source_table: str) -> dict:
        """
        Normalize a tender that comes directly from test data (not from Supabase).
        
        Args:
            tender: Dictionary containing tender data
            source_table: The source table name
            
        Returns:
            Dictionary with normalized data
        """
        # Convert dict to RawTender model
        raw_tender = RawTender(
            id=tender.get("id", f"test_{source_table}_{int(time.time())}"),
            source_table=source_table,
            **tender
        )
        
        # Use the synchronous normalize_tender method
        return self.normalize_tender_sync(raw_tender, source_table)
    
    def _parse_fields_directly(self, tender: RawTender) -> Dict[str, Any]:
        """
        Directly parse fields from the raw tender data as a fallback.
        """
        source_table = tender.source_table
        
        # Default country values based on source
        default_country = {
            "sam_gov": "United States",
            "ted_eu": "European Union",
            "ungm": "International",
            "aiib": "International",
            "wb": tender.country or "International",
            "adb": tender.country or "Asia",
            "iadb": tender.country or "International",
            "afdb": tender.country or "Africa",
            "afd_tenders": tender.country or "France"
        }.get(source_table, tender.country or "")
        
        # For sources that need title generation
        title = tender.title or ""
        if not title:
            if source_table == "adb":
                title = f"ADB Tender: {tender.id}"
            elif source_table == "iadb":
                title = f"IADB Project: {tender.id}"
        
        # Handle date parsing
        publication_date = tender.publication_date or ""
        if source_table == "afd_tenders" and publication_date:
            try:
                # Try to parse the date string in the format "Feb 6, 2025"
                date_obj = datetime.strptime(publication_date, "%b %d, %Y")
                publication_date = date_obj.isoformat()
            except ValueError:
                publication_date = datetime.now().isoformat()
        
        # Handle "Unknown" dates for AFDB
        if source_table == "afdb" and publication_date == "Unknown":
            publication_date = datetime.now().isoformat()
        
        return {
            "id": tender.id,
            "source_table": source_table,
            "source_id": tender.id, 
            "title": title,
            "description": tender.description or "",
            "country": default_country,
            "organization_name": tender.organization_name or "",
            "publication_date": publication_date,
            "url": tender.url or "",
            "normalized_by": "direct_parsing"
        }

    def log_performance_stats(self):
        """Log performance statistics for the normalizer."""
        if hasattr(self, 'performance_stats'):
            stats = self.performance_stats
            total = stats.get("total_processed", 0)
            
            if total > 0:
                # Make sure we have a default value of 0 if success_rate is not set
                success_rate = stats.get("success_rate", 0.0)
                avg_time = stats.get("normalization_time", 0) / total if total > 0 else 0
                
                logging.info(f"Total processed: {total}")
                logging.info(f"Success rate: {success_rate:.1f}%")
                logging.info(f"Average processing time: {avg_time:.2f}s")
        else:
            logging.info("No performance stats available.")


# Create a singleton instance
normalizer = TenderNormalizer() 