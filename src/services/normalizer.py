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

from langdetect import detect, LangDetectException
from deep_translator import GoogleTranslator
from pydantic import BaseModel, Field, ValidationError, model_validator
from pydantic_ai import Agent, RunContext

from ..config import normalizer_config, settings
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
    """
    Input for the LLM-based tender normalization.
    """
    
    raw_tender: Dict[str, Any] = Field(
        ..., description="The raw tender data to be normalized"
    )
    source_table: str = Field(
        ..., description="The source table name (e.g., 'sam_gov', 'wb')"
    )


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

    def __init__(self) -> None:
        """Initialize the tender normalizer."""
        # Set OpenAI API key in environment if not already set
        if "OPENAI_API_KEY" not in os.environ:
            if "OPENAI_KEY" in os.environ:
                os.environ["OPENAI_API_KEY"] = os.environ["OPENAI_KEY"]
            elif settings.openai_api_key.get_secret_value():
                os.environ["OPENAI_API_KEY"] = settings.openai_api_key.get_secret_value()
                
        # Set up the agent for normalization
        try:
            self.agent = Agent(
                settings.openai_model,
                result_type=NormalizationOutput,
                system_prompt=self._get_system_prompt(),
            )
        except TypeError:
            # Fallback if the Agent constructor signature is different
            self.agent = Agent(
                result_type=NormalizationOutput,
                description=self._get_system_prompt(),
            )
        
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

    async def _normalize_with_llm(self, tender: RawTender) -> NormalizationResult:
        """
        Normalize a tender using the LLM.
        
        Args:
            tender: The raw tender to normalize
            
        Returns:
            Normalization result
        """
        start_time = time.time()
        tender_dict = tender.model_dump(exclude={"source_data"})
        fields_before = self._count_non_empty_fields(tender_dict)
        
        try:
            # Prepare input for LLM normalization
            input_data = NormalizationInput(
                raw_tender=tender_dict,
                source_table=tender.source_table,
            )
            
            # Create a run context (will be passed to the agent)
            try:
                # Try newer pydantic-ai API
                context = RunContext(model="mistral")
            except TypeError:
                # Try older pydantic-ai API
                try:
                    context = RunContext(
                        deps={},
                        model="mistral",
                        usage={},
                        prompt=self._get_system_prompt()
                    )
                except Exception as e:
                    logger.warning(f"Error creating RunContext with deps/model/usage/prompt: {str(e)}")
                    # Last resort fallback
                    try:
                        context = RunContext()
                    except Exception as e:
                        logger.warning(f"Error creating basic RunContext: {str(e)}")
                        context = None
            
            # Run the normalization
            if context is not None:
                try:
                    result = await self.agent.run(input_data, context=context)
                except TypeError:
                    # Try older API without context parameter
                    result = await self.agent.run(input_data)
            
            output = NormalizationOutput.model_validate(result)
            
            # Save performance data
            processing_time = time.time() - start_time
            processing_time_ms = int(processing_time * 1000)
            
            # Ensure required fields and correct types
            tender_data = output.tender
            # Add fields that might be missing
            tender_data["id"] = tender.id
            tender_data["source_table"] = tender.source_table
            tender_data["normalized_by"] = "LLM-Mistral"
            tender_data["normalized_method"] = "llm"
            tender_data["normalized_at"] = datetime.utcnow().isoformat()
            tender_data["processing_time_ms"] = processing_time_ms
            
            # Apply post-processing to improve quality
            if "title" in tender_data and tender_data["title"]:
                tender_data["title"] = format_title(tender_data["title"])
            
            # Extract URLs from description and other text fields if not already found
            if "url" not in tender_data or not tender_data["url"]:
                # Check description field for URLs
                if "description" in tender_data and tender_data["description"]:
                    urls = extract_urls_from_text(tender_data["description"])
                    if urls:
                        tender_data["url"] = urls[0]  # Use the first URL
                        if len(urls) > 1:
                            tender_data.setdefault("document_links", [])
                            for url in urls[1:]:
                                tender_data["document_links"].append({"url": url})
                
                # Check all fields for URLs if we still don't have one
                if "url" not in tender_data or not tender_data["url"]:
                    for field_name, field_value in tender_dict.items():
                        if isinstance(field_value, str) and field_name not in ["id", "source_id"]:
                            urls = extract_urls_from_text(field_value)
                            if urls:
                                tender_data["url"] = urls[0]
                                break
            
            # Extract contact information from description if not already found
            has_contact_info = "contact_name" in tender_data or "contact_email" in tender_data or "contact_phone" in tender_data
            if not has_contact_info and "description" in tender_data and tender_data["description"]:
                emails = extract_emails_from_text(tender_data["description"])
                if emails:
                    tender_data["contact_email"] = emails[0]
            
            # Handle language detection and translation
            language = None
            
            # Detect language from title and description
            if "title" in tender_data and tender_data["title"]:
                language = detect_language(tender_data["title"])
                tender_data["language"] = language
                
                # Translate title if not in English
                if language != "en":
                    translated_title = translate_to_english(tender_data["title"], language)
                    if translated_title != tender_data["title"]:
                        tender_data["title_english"] = translated_title
            
            # Translate description if not in English
            if "description" in tender_data and tender_data["description"]:
                desc_language = detect_language(tender_data["description"])
                if not language:
                    language = desc_language
                    tender_data["language"] = language
                
                if desc_language != "en":
                    translated_desc = translate_to_english(tender_data["description"], desc_language)
                    if translated_desc != tender_data["description"]:
                        tender_data["description_english"] = translated_desc
            
            # Organization name translation if needed
            if "organization_name" in tender_data and tender_data["organization_name"]:
                org_language = detect_language(tender_data["organization_name"])
                if org_language != "en":
                    translated_org = translate_to_english(tender_data["organization_name"], org_language)
                    if translated_org != tender_data["organization_name"]:
                        tender_data["organization_name_english"] = translated_org
            
            # Handle dates
            self._infer_status_from_dates(tender_data)
            
            # Ensure all critical fields are present
            self._ensure_critical_fields(tender_data, tender)
            
            # Log quality check data
            log_quality_check(tender.id, tender.source_table, tender_dict, tender_data)
            
            # Create the normalized tender
            normalized_tender = NormalizedTender.model_validate(tender_data)
            
            # Calculate fields and improvement
            fields_after = self._count_non_empty_fields(normalized_tender.model_dump())
            improvement = (
                ((fields_after - fields_before) / fields_before) * 100
                if fields_before > 0
                else 0
            )
            
            # Update stats
            result = NormalizationResult(
                tender_id=tender.id,
                source_table=tender.source_table,
                success=True,
                normalized_tender=normalized_tender,
                error=None,
                processing_time=processing_time,
                method_used="llm",
                fields_before=fields_before,
                fields_after=fields_after,
                improvement_percentage=improvement,
            )
            
            self._update_performance_stats(result)
            return result
            
        except Exception as e:
            logger.error(
                f"LLM normalization failed for {tender.id} from {tender.source_table}: {str(e)}"
            )
            
            # Fallback to direct parsing if LLM fails
            return await self._normalize_with_fallback(
                tender, f"LLM normalization failed: {str(e)}", start_time
            )

    async def _normalize_with_fallback(
        self, tender: RawTender, error: str = "Fallback used", start_time: Optional[float] = None
    ) -> NormalizationResult:
        """
        Normalize a tender using rule-based fallback when LLM fails.
        
        Args:
            tender: The raw tender to normalize
            error: The error message from the LLM normalization attempt
            start_time: Optional start time for performance tracking
            
        Returns:
            Normalization result
        """
        if start_time is None:
            start_time = time.time()
            
        tender_dict = tender.model_dump(exclude={"source_data"})
        fields_before = self._count_non_empty_fields(tender_dict)
        
        try:
            # Basic normalized data with metadata
            normalized_data = {
                "id": tender.id,
                "source_id": str(tender.source_id),
                "source_table": tender.source_table,
                "normalized_by": "rule-based-fallback",
                "normalized_method": "fallback",
                "fallback_reason": error,
                "normalized_at": datetime.utcnow().isoformat(),
                "processing_time_ms": int((time.time() - start_time) * 1000),
            }
            
            # Map direct fields
            direct_field_mappings = [
                "title", "description", "country", "country_code", "location",
                "organization_name", "organization_id", "title_english",
                "description_english", "organization_name_english", "language",
                "buyer", "project_name", "project_id", "project_number", "sector",
                "reference_number", "notice_id", "procurement_method", "tender_type",
                "status", "publication_date", "deadline_date", "url", "estimated_value",
                "currency", "city", "contact_name", "contact_email", "contact_phone", 
                "contact_address",
            ]
            
            # Add source-specific field mappings
            source_specific_mappings = {
                "sam_gov": {
                    "title": "opportunity_title",
                    "description": "description",
                    "country": "place_of_performance.country_name" if hasattr(tender, "place_of_performance") else None,
                    "organization_name": "organization_name",
                    "buyer": "organization_name",
                    "notice_id": "notice_id",
                    "reference_number": "solicitation_number",
                    "procurement_method": "competitive_procedures_code",
                    "city": "place_of_performance.city_name" if hasattr(tender, "place_of_performance") else None,
                    "contact_name": "point_of_contact",
                    "contact_email": "point_of_contact_email",
                    "contact_phone": "point_of_contact_phone",
                    "publication_date": "posted_date",
                    "deadline_date": "response_deadline",
                    "status": "archive_type", 
                    "url": "solicitation_link",
                    "source_id": "opportunity_id",
                },
                "wb": {
                    "title": "title",
                    "description": "description",
                    "country": "country",
                    "organization_name": "contact_organization",
                    "project_name": "project_name",
                    "project_id": "project_id",
                    "sector": "sector",
                    "notice_id": "notice_no",
                    "procurement_method": "procurement_method",
                    "contact_name": "contact_person",
                    "contact_email": "contact_email",
                    "contact_phone": "contact_phone",
                    "publication_date": "published",
                    "deadline_date": "deadline",
                    "estimated_value": "contract_value",
                    "currency": "contract_currency",
                    "url": "link",
                    "source_id": "id",
                },
                "adb": {
                    "title": "notice_title",
                    "description": "description",
                    "country": "country",
                    "project_name": "project_name",
                    "project_id": "project_id",
                    "project_number": "project_number",
                    "sector": "sector",
                    "notice_id": "notice_number",
                    "procurement_method": "procurement_method",
                    "buyer": "executing_agency",
                    "publication_date": "published_date",
                    "deadline_date": "closing_date",
                    "url": "link",
                    "source_id": "id",
                },
                "ted_eu": {
                    "title": "title",
                    "description": "summary",
                    "organization_name": "organisation_name",
                    "buyer": "organisation_name",
                    "notice_id": "notice_id",
                    "reference_number": "reference_number",
                    "procurement_method": "procedure_type",
                    "city": "town",
                    "contact_name": "contact_officer_name",
                    "contact_email": "contact_email",
                    "contact_phone": "contact_telephone",
                    "contact_address": "contact_address",
                    "publication_date": "document_sent_date",
                    "deadline_date": "deadline",
                    "estimated_value": "value_magnitude",
                    "currency": "value_currency",
                    "url": "notice_url",
                    "source_id": "id",
                },
                "ungm": {
                    "title": "title",
                    "description": "description",
                    "country": "beneficiary_countries",
                    "organization_name": "un_organization",
                    "reference_number": "reference",
                    "deadline_date": "deadline",
                    "notice_id": "notice_id",
                    "buyer": "un_organization",
                    "procurement_method": "procurement_method",
                    "contact_name": "contact_name",
                    "contact_email": "contact_email",
                    "publication_date": "published_date",
                    "url": "link",
                    "source_id": "id",
                },
                "afd_tenders": {
                    "title": "notice_title",
                    "description": "notice_content",
                    "country": "country",
                    "organization_name": "buyer",
                    "buyer": "buyer",
                    "reference_number": "reference",
                    "notice_id": "notice_id",
                    "sector": "sector",
                    "publication_date": "launch_date",
                    "deadline_date": "closure_date",
                    "url": "notice_url",
                    "source_id": "id",
                },
                "iadb": {
                    "title": "notice_title",
                    "description": "project_name",
                    "country": "country",
                    "project_name": "project_name",
                    "project_id": "project_number",
                    "notice_id": "notice_id",
                    "buyer": "borrower",
                    "sector": "sector",
                    "publication_date": "published_date",
                    "deadline_date": "deadline_date",
                    "url": "link",
                    "source_id": "id",
                },
                "afdb": {
                    "title": "title",
                    "description": "description",
                    "country": "country",
                    "project_name": "project_name",
                    "project_id": "project_number",
                    "sector": "sector",
                    "notice_id": "reference_number",
                    "reference_number": "reference_number",
                    "buyer": "borrower",
                    "publication_date": "publication_date",
                    "deadline_date": "deadline_date",
                    "url": "tender_url",
                    "source_id": "id",
                },
                "aiib": {
                    "title": "project_notice",
                    "description": "project_notice",
                    "country": "member",
                    "project_name": "project_name",
                    "project_id": "project_id",
                    "sector": "sector",
                    "notice_id": "notice_id",
                    "buyer": "borrower",
                    "publication_date": "publication_date",
                    "deadline_date": "deadline",
                    "url": "link",
                    "source_id": "id",
                }
            }
            
            # Define additional URL field names to check - prioritize source-specific URL fields
            source_url_fields = self._get_url_fields_for_source(tender.source_table)
            url_field_names = source_url_fields + [
                "url", "link", "web_link", "notice_url", "opportunity_link", "solicitation_link", 
                "tender_url", "project_url", "document_url", "listing_url", "external_url"
            ]
            
            # Define additional contact field names to check
            contact_field_names = {
                "name": ["contact_name", "point_of_contact", "contact_person", "contact", "focal_point", 
                         "officer_name", "responsible_officer", "respondent"],
                "email": ["contact_email", "email", "point_of_contact_email", "officer_email", "email_address"],
                "phone": ["contact_phone", "phone", "phone_number", "telephone", "contact_telephone", 
                          "tel", "mobile", "officer_phone"],
                "address": ["contact_address", "address", "postal_address", "mailing_address", "office_address"]
            }
            
            # Apply source-specific mappings first
            if tender.source_table in source_specific_mappings:
                mappings = source_specific_mappings[tender.source_table]
                for norm_field, source_field in mappings.items():
                    if source_field:
                        # Handle nested fields with dot notation (e.g., "place_of_performance.city_name")
                        if "." in source_field:
                            parts = source_field.split(".")
                            value = tender_dict
                            for part in parts:
                                if isinstance(value, dict) and part in value:
                                    value = value[part]
                                else:
                                    value = None
                                    break
                        else:
                            # Try to get from attributes first
                            value = getattr(tender, source_field, None)
                            # If not found, try to get from source_data
                            if value is None and hasattr(tender, "source_data") and tender.source_data:
                                value = tender.source_data.get(source_field)
                        
                        if value is not None:
                            # For source_id, convert to string
                            if norm_field == "source_id":
                                normalized_data[norm_field] = str(value)
                            # For other fields, only add if not empty
                            elif str(value).strip():
                                normalized_data[norm_field] = value
            
            # Then apply direct mappings for any fields not already mapped
            for field in direct_field_mappings:
                if field not in normalized_data:
                    # Try to get from attributes first
                    value = getattr(tender, field, None)
                    # If not found, try to get from source_data
                    if value is None and hasattr(tender, "source_data") and tender.source_data:
                        value = tender.source_data.get(field)
                    
                    if value is not None:
                        # For source_id, convert to string
                        if field == "source_id":
                            normalized_data[field] = str(value)
                        # For other fields, only add if not empty
                        elif str(value).strip():
                            normalized_data[field] = value
            
            # Apply post-processing to improve quality
            if "title" in normalized_data and normalized_data["title"]:
                normalized_data["title"] = format_title(normalized_data["title"])
            
            # Handle language detection and translation
            language = None
            
            # Detect language from title and description
            if "title" in normalized_data and normalized_data["title"]:
                language = detect_language(normalized_data["title"])
                normalized_data["language"] = language
                
                # Translate title if not in English
                if language != "en":
                    translated_title = translate_to_english(normalized_data["title"], language)
                    if translated_title != normalized_data["title"]:
                        normalized_data["title_english"] = translated_title
            
            # Translate description if not in English
            if "description" in normalized_data and normalized_data["description"]:
                desc_language = detect_language(normalized_data["description"])
                if not language:
                    language = desc_language
                    normalized_data["language"] = language
                
                if desc_language != "en":
                    translated_desc = translate_to_english(normalized_data["description"], desc_language)
                    if translated_desc != normalized_data["description"]:
                        normalized_data["description_english"] = translated_desc
            
            # Organization name translation if needed
            if "organization_name" in normalized_data and normalized_data["organization_name"]:
                org_language = detect_language(normalized_data["organization_name"])
                if org_language != "en":
                    translated_org = translate_to_english(normalized_data["organization_name"], org_language)
                    if translated_org != normalized_data["organization_name"]:
                        normalized_data["organization_name_english"] = translated_org
            
            # Handle URL fields - Check all possible URL field names, but prioritize source-specific ones
            url_found = False
            
            # First check the source-specific primary URL field
            primary_url_field = source_url_fields[0] if source_url_fields else None
            
            if primary_url_field:
                # Check direct attribute
                url_value = getattr(tender, primary_url_field, None)
                if url_value is None and hasattr(tender, "source_data") and tender.source_data:
                    url_value = tender.source_data.get(primary_url_field)
                
                if url_value and str(url_value).strip():
                    normalized_data["url"] = str(url_value).strip()
                    url_found = True
                    logger.info(f"Found primary URL from field {primary_url_field}: {url_value}")
            
            # If no primary URL found, check other URL fields
            if not url_found:
                for url_field in url_field_names:
                    if url_field in normalized_data and normalized_data[url_field]:
                        # If we already have a primary URL but found another one, add it to document_links
                        if url_found and url_field != "url":
                            normalized_data.setdefault("document_links", [])
                            normalized_data["document_links"].append({"url": normalized_data[url_field]})
                        else:
                            # Set as primary URL if we don't have one yet
                            normalized_data["url"] = normalized_data[url_field]
                            url_found = True
                    
                    # Also check tender attributes and source_data
                    if not url_found or url_field != "url":
                        url_value = getattr(tender, url_field, None)
                        if url_value is None and hasattr(tender, "source_data") and tender.source_data:
                            url_value = tender.source_data.get(url_field)
                        
                        if url_value and str(url_value).strip():
                            if url_found and url_field != "url":
                                normalized_data.setdefault("document_links", [])
                                normalized_data["document_links"].append({"url": url_value})
                            else:
                                normalized_data["url"] = url_value
                                url_found = True
            
            # Extract URLs from description and other text fields only as a fallback
            if not url_found and "description" in normalized_data and normalized_data["description"]:
                urls = extract_urls_from_text(normalized_data["description"])
                if urls:
                    normalized_data["url"] = urls[0]  # Use the first URL
                    url_found = True
                    logger.info(f"Extracted URL from description: {urls[0]}")
                    
                    if len(urls) > 1:
                        normalized_data.setdefault("document_links", [])
                        for url in urls[1:]:
                            normalized_data["document_links"].append({"url": url})
            
            # Log if no URL was found
            if not url_found:
                logger.warning(f"No URL found for tender {tender.id} from {tender.source_table}")
            
            # Handle document links
            documents = []
            doc_links = getattr(tender, "documents", None) or getattr(tender, "document_links", None)
            
            if doc_links:
                if isinstance(doc_links, list):
                    for doc in doc_links:
                        if isinstance(doc, dict):
                            documents.append(doc)
                        elif isinstance(doc, str) and (doc.startswith("http://") or doc.startswith("https://")):
                            documents.append({"url": doc})
                elif isinstance(doc_links, dict):
                    documents.append(doc_links)
                elif isinstance(doc_links, str) and (doc_links.startswith("http://") or doc_links.startswith("https://")):
                    documents.append({"url": doc_links})
            
            if documents:
                normalized_data["document_links"] = documents
            
            # Contact information - check all possible field names
            contact_info = {}
            
            # Try to extract contact info from all available fields using the mappings
            for contact_type, field_names in contact_field_names.items():
                for field_name in field_names:
                    # Try to get from normalized_data first
                    value = normalized_data.get(field_name)
                    
                    # If not found, try attributes
                    if not value:
                        value = getattr(tender, field_name, None)
                    
                    # If not found, try source_data
                    if not value and hasattr(tender, "source_data") and tender.source_data:
                        value = tender.source_data.get(field_name)
                    
                    if value and str(value).strip():
                        contact_info[contact_type] = value
                        # Also set the specific contact field directly
                        if contact_type == "name":
                            normalized_data["contact_name"] = value
                        elif contact_type == "email":
                            normalized_data["contact_email"] = value
                        elif contact_type == "phone":
                            normalized_data["contact_phone"] = value
                        elif contact_type == "address":
                            normalized_data["contact_address"] = value
                        break
            
            # If we still don't have contact email, try to extract from description
            if "email" not in contact_info and "description" in normalized_data:
                emails = extract_emails_from_text(normalized_data["description"])
                if emails:
                    contact_info["email"] = emails[0]
                    normalized_data["contact_email"] = emails[0]
                    logger.info(f"Extracted email from description: {emails[0]}")
            
            # If we have any contact info, add it in both formats
            if contact_info:
                normalized_data["contact"] = contact_info
            
            # Dates
            for date_field in ["publication_date", "deadline_date"]:
                date_value = getattr(tender, date_field, None)
                if date_value is None and hasattr(tender, "source_data") and tender.source_data:
                    date_value = tender.source_data.get(date_field)
                
                if date_value:
                    # Skip values like "Unknown", "TBD", etc.
                    if isinstance(date_value, str) and (
                        date_value.lower() in ["unknown", "tbd", "to be determined", "n/a", "not available"]
                    ):
                        continue
                        
                    if isinstance(date_value, (datetime, date)):
                        normalized_data[date_field] = date_value
                    elif isinstance(date_value, str):
                        parsed_date = self._parse_date(date_value)
                        if parsed_date:
                            normalized_data[date_field] = parsed_date
            
            # Infer status from dates
            self._infer_status_from_dates(normalized_data)
            
            # Ensure all critical fields are present
            self._ensure_critical_fields(normalized_data, tender)
            
            # Log quality check
            log_quality_check(tender.id, tender.source_table, tender_dict, normalized_data)
            
            # Create the normalized tender
            normalized_tender = NormalizedTender.model_validate(normalized_data)
            
            # Calculate fields and improvement
            fields_after = self._count_non_empty_fields(normalized_tender.model_dump())
            processing_time = time.time() - start_time
            improvement = (
                ((fields_after - fields_before) / fields_before) * 100
                if fields_before > 0
                else 0
            )
            
            # After all mappings are done, standardize fields that need normalization
            # Standardize tender_type to match our schema
            self._map_standard_tender_type(normalized_data, tender)
            
            # Create the result
            result = NormalizationResult(
                tender_id=tender.id,
                source_table=tender.source_table,
                success=True,
                normalized_tender=normalized_tender,
                error=None,
                processing_time=processing_time,
                method_used="fallback",
                fields_before=fields_before,
                fields_after=fields_after,
                improvement_percentage=improvement,
            )
            
            self._update_performance_stats(result)
            return result
            
        except Exception as e:
            logger.error(
                f"Fallback normalization failed for {tender.id} from {tender.source_table}: {str(e)}"
            )
            processing_time = time.time() - start_time
            
            return NormalizationResult(
                tender_id=tender.id,
                source_table=tender.source_table,
                success=False,
                normalized_tender=None,
                error=f"Both LLM and fallback normalization failed: {str(e)}",
                processing_time=processing_time,
                method_used="failed",
                fields_before=fields_before,
                fields_after=0,
                improvement_percentage=0,
            )

    def _parse_date(self, date_str: Optional[str]) -> Optional[str]:
        """Parse a date string into ISO format or return None if invalid."""
        if not date_str or date_str == "Unknown" or date_str.lower() == "unknown":
            return None
        
        try:
            # Try different date formats
            for date_format in [
                "%Y-%m-%d",
                "%Y/%m/%d",
                "%d-%m-%Y",
                "%d/%m/%Y",
                "%m-%d-%Y",
                "%m/%d/%Y",
                "%b %d, %Y",
                "%B %d, %Y",
                "%d %b %Y",
                "%d %B %Y",
                "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%dT%H:%M:%S.%f",
                "%Y-%m-%dT%H:%M:%SZ",
                "%Y-%m-%dT%H:%M:%S.%fZ",
            ]:
                try:
                    dt = datetime.strptime(date_str, date_format)
                    return dt.date().isoformat()
                except ValueError:
                    continue
            
            # If none of the formats work, try dateutil parser as last resort
            from dateutil import parser
            dt = parser.parse(date_str)
            return dt.date().isoformat()
        except Exception as e:
            logger.warning(f"Failed to parse date: {date_str} - {str(e)}")
            return None

    def _infer_status_from_dates(self, tender_data: Dict[str, Any]) -> None:
        """
        Infer tender status from publication and deadline dates if status is not available.
        
        Args:
            tender_data: The tender data dictionary
        """
        # Skip if status is already set and not "unknown"
        if tender_data.get("status") and tender_data["status"].lower() != "unknown":
            return
        
        # Parse dates if they're strings
        publication_date = None
        deadline_date = None
        
        if "publication_date" in tender_data:
            if isinstance(tender_data["publication_date"], str):
                publication_date = self._parse_date(tender_data["publication_date"])
                # Update the field with parsed value
                tender_data["publication_date"] = publication_date
            elif isinstance(tender_data["publication_date"], (datetime, date)):
                publication_date = tender_data["publication_date"].isoformat()
        
        if "deadline_date" in tender_data:
            if isinstance(tender_data["deadline_date"], str):
                deadline_date = self._parse_date(tender_data["deadline_date"])
                # Update the field with parsed value
                tender_data["deadline_date"] = deadline_date
            elif isinstance(tender_data["deadline_date"], (datetime, date)):
                deadline_date = tender_data["deadline_date"].isoformat()
        
        # Infer status based on dates
        today = date.today().isoformat()
        
        if deadline_date:
            if deadline_date < today:
                tender_data["status"] = "closed"
            else:
                tender_data["status"] = "active"
        elif publication_date:
            if publication_date > today:
                tender_data["status"] = "upcoming"
            else:
                # Default to active if we only have publication date
                tender_data["status"] = "active"
        else:
            # No useful date information
            tender_data["status"] = "unknown"

    def _ensure_critical_fields(self, normalized_data: Dict[str, Any], tender: RawTender) -> None:
        """
        Ensure all critical fields are present in the normalized data.
        Generate fallback values if necessary.
        """
        # Title is required
        if not normalized_data.get("title"):
            # Generate a title from description if available
            if tender.description:
                # Use the first sentence or up to 100 chars
                desc = tender.description
                if len(desc) > 100:
                    title = desc[:100] + "..."
                else:
                    title = desc
                normalized_data["title"] = title
            else:
                # Use a generic title with the ID
                normalized_data["title"] = f"Tender {tender.id} from {tender.source_table}"
        
        # Description is required
        if not normalized_data.get("description"):
            # Use title as description if available
            if normalized_data.get("title"):
                normalized_data["description"] = normalized_data["title"]
            else:
                # Use a generic description
                normalized_data["description"] = f"Tender {tender.id} from {tender.source_table}"
        
        # Country is required
        if not normalized_data.get("country"):
            # Try to extract from other fields
            for field in ["location", "organization_name", "title", "description"]:
                value = getattr(tender, field, None)
                if value and isinstance(value, str):
                    # Very simple country extraction - would need improvement
                    common_countries = [
                        "United States", "Canada", "UK", "Australia", "India", 
                        "France", "Germany", "Japan", "China", "Brazil"
                    ]
                    for country in common_countries:
                        if country.lower() in value.lower():
                            normalized_data["country"] = country
                            break
                    
                    if normalized_data.get("country"):
                        break
            
            # If still no country, use a default based on source
            if not normalized_data.get("country"):
                source_country_map = {
                    "sam_gov": "United States",
                    "wb": "Global",
                    "adb": "Asia",
                    "ted_eu": "European Union",
                    "ungm": "Global",
                    "afd_tenders": "Global",
                    "iadb": "Latin America",
                    "afdb": "Africa",
                    "aiib": "Asia",
                }
                normalized_data["country"] = source_country_map.get(
                    tender.source_table, "Unknown"
                )
        
        # Ensure source_id is a string and is always present
        if not normalized_data.get("source_id"):
            # Default to tender.id if source_id is not set
            normalized_data["source_id"] = str(tender.id)
        elif not isinstance(normalized_data["source_id"], str):
            # Convert to string if not already a string
            normalized_data["source_id"] = str(normalized_data["source_id"])

    def _count_non_empty_fields(self, data: Dict[str, Any]) -> int:
        """Count the number of non-empty fields in a dictionary."""
        count = 0
        for key, value in data.items():
            if value is not None:
                if isinstance(value, str) and not value.strip():
                    continue
                if isinstance(value, list) and not value:
                    continue
                if isinstance(value, dict) and not value:
                    continue
                count += 1
        return count

    def _update_performance_stats(self, result: NormalizationResult) -> None:
        """Update performance statistics."""
        self.performance_stats["total_processed"] += 1
        
        if result.method_used == "llm":
            self.performance_stats["llm_used"] += 1
        elif result.method_used == "fallback":
            self.performance_stats["fallback_used"] += 1
        
        self.performance_stats["total_processing_time"] += result.processing_time
        self.performance_stats["avg_processing_time"] = (
            self.performance_stats["total_processing_time"] / 
            self.performance_stats["total_processed"]
        )
        
        self.performance_stats["success_rate"] = (
            (self.performance_stats["total_processed"] - 
             self.performance_stats.get("failed", 0)) / 
            self.performance_stats["total_processed"]
        ) * 100
        
        # Track by source
        source = result.source_table
        if source not in self.performance_stats["by_source"]:
            self.performance_stats["by_source"][source] = {
                "total": 0,
                "llm_used": 0,
                "fallback_used": 0,
                "success_rate": 0,
                "avg_time": 0,
            }
        
        source_stats = self.performance_stats["by_source"][source]
        source_stats["total"] += 1
        
        if result.method_used == "llm":
            source_stats["llm_used"] += 1
        elif result.method_used == "fallback":
            source_stats["fallback_used"] += 1
        
        source_stats["avg_time"] = (
            (source_stats["avg_time"] * (source_stats["total"] - 1) + 
             result.processing_time) / source_stats["total"]
        )
        
        source_stats["success_rate"] = (
            (source_stats["total"] - source_stats.get("failed", 0)) / 
            source_stats["total"]
        ) * 100

    def log_performance_stats(self) -> None:
        """Log performance statistics."""
        logger.info("------- Normalization Performance Statistics -------")
        logger.info(f"Total tenders processed: {self.performance_stats['total_processed']}")
        logger.info(f"LLM normalization used: {self.performance_stats['llm_used']} "
                    f"({self.performance_stats['llm_used'] / self.performance_stats['total_processed'] * 100:.1f}%)")
        logger.info(f"Fallback normalization used: {self.performance_stats['fallback_used']} "
                    f"({self.performance_stats['fallback_used'] / self.performance_stats['total_processed'] * 100:.1f}%)")
        logger.info(f"Success rate: {self.performance_stats['success_rate']:.1f}%")
        logger.info(f"Average processing time: {self.performance_stats['avg_processing_time']:.2f}s")
        logger.info(f"Total processing time: {self.performance_stats['total_processing_time']:.1f}s")
        
        logger.info("------- By Source -------")
        for source, stats in self.performance_stats["by_source"].items():
            logger.info(f"{source}: {stats['total']} tenders, "
                        f"{stats['llm_used'] / stats['total'] * 100:.1f}% LLM, "
                        f"{stats['success_rate']:.1f}% success, "
                        f"{stats['avg_time']:.2f}s avg time")
        logger.info("----------------------------------------------------")

    async def normalize_tender(self, tender: RawTender) -> NormalizationResult:
        """
        Normalize a tender, using LLM if appropriate or falling back to rule-based methods.
        
        Args:
            tender: The raw tender to normalize
            
        Returns:
            Normalization result
        """
        # Determine if we should use LLM
        should_use_llm, reason = self._should_use_llm(tender)
        logger.info(f"Processing tender {tender.id} from {tender.source_table} - "
                   f"LLM: {should_use_llm} ({reason})")
        
        # Normalize using appropriate method
        if should_use_llm and settings.use_llm_normalization:
            result = await self._normalize_with_llm(tender)
        else:
            result = await self._normalize_with_fallback(
                tender, error=f"LLM skipped: {reason}"
            )
        
        # Update stats
        self._update_performance_stats(result)
        
        # Log success or failure
        if result.success:
            logger.info(
                f"Normalized tender {tender.id} from {tender.source_table} using {result.method_used} "
                f"in {result.processing_time:.2f}s - Fields: {result.fields_before} → {result.fields_after} "
                f"({result.improvement_percentage:.2f}% improvement)"
            )
        else:
            logger.error(
                f"Failed to normalize tender {tender.id} from {tender.source_table}: {result.error}"
            )
        
        return result

    async def normalize_tenders(self, tenders: List[RawTender]) -> List[NormalizationResult]:
        """
        Normalize a batch of tenders concurrently.
        
        Args:
            tenders: List of raw tenders to normalize
            
        Returns:
            List of normalization results
        """
        # Process tenders with concurrency limit
        semaphore = asyncio.Semaphore(settings.concurrent_requests)
        
        async def _normalize_with_semaphore(tender: RawTender) -> NormalizationResult:
            async with semaphore:
                return await self.normalize_tender(tender)
        
        # Create tasks for all tenders
        tasks = [_normalize_with_semaphore(tender) for tender in tenders]
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results to handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                # Handle exception
                logger.error(f"Error normalizing tender: {str(result)}")
                processed_results.append(
                    NormalizationResult(
                        tender_id=tenders[i].id,
                        source_table=tenders[i].source_table,
                        success=False,
                        normalized_tender=None,
                        error=str(result),
                        processing_time=0.0,
                        method_used="error",
                        fields_before=0,
                        fields_after=0,
                        improvement_percentage=0.0,
                    )
                )
            else:
                processed_results.append(result)
        
        # Log performance stats if we've processed enough tenders
        if self.performance_stats["total_processed"] % 10 == 0:
            self.log_performance_stats()
        
        return processed_results

    async def normalize_test_batch(self, tenders_by_source: Dict[str, List[RawTender]]) -> Dict[str, List[NormalizationResult]]:
        """
        Normalize a test batch of tenders with extensive logging for quality improvement.
        
        This function processes a small number of tenders from each source table
        with detailed logging to help identify and fix normalization issues.
        
        Args:
            tenders_by_source: Dictionary mapping source tables to lists of tenders
            
        Returns:
            Dictionary mapping source tables to lists of normalization results
        """
        results_by_source = {}
        
        for source_table, tenders in tenders_by_source.items():
            logger.info(f"==== PROCESSING TEST BATCH FOR {source_table} ====")
            logger.info(f"Test batch contains {len(tenders)} tenders")
            
            # Process each tender individually for detailed logging
            source_results = []
            for i, tender in enumerate(tenders):
                logger.info(f"[{source_table}] Processing test tender {i+1}/{len(tenders)}: {tender.id}")
                
                # Log raw tender data
                logger.info(f"[{source_table}:{tender.id}] RAW TENDER DATA:")
                tender_dict = tender.model_dump()
                
                # Log source-specific URL fields
                url_fields = self._get_url_fields_for_source(source_table)
                url_values = {}
                for field in url_fields:
                    if field in tender_dict:
                        url_values[field] = tender_dict[field]
                    elif hasattr(tender, "source_data") and tender.source_data and field in tender.source_data:
                        url_values[field] = tender.source_data[field]
                
                if url_values:
                    logger.info(f"[{source_table}:{tender.id}] URL FIELDS: {url_values}")
                else:
                    logger.info(f"[{source_table}:{tender.id}] NO URL FIELDS FOUND")
                
                # Normalize the tender
                result = await self.normalize_tender(tender)
                source_results.append(result)
                
                # Log result
                if result.success and result.normalized_tender:
                    logger.info(f"[{source_table}:{tender.id}] NORMALIZATION SUCCESSFUL ({result.method_used})")
                    
                    # Log URL extraction results
                    normalized_data = result.normalized_tender.model_dump()
                    if "url" in normalized_data and normalized_data["url"]:
                        logger.info(f"[{source_table}:{tender.id}] EXTRACTED URL: {normalized_data['url']}")
                    else:
                        logger.info(f"[{source_table}:{tender.id}] NO URL EXTRACTED")
                    
                    # Log contact extraction results
                    contact_fields = ["contact_name", "contact_email", "contact_phone", "contact_address"]
                    extracted_contacts = {f: normalized_data.get(f) for f in contact_fields if f in normalized_data and normalized_data.get(f)}
                    if extracted_contacts:
                        logger.info(f"[{source_table}:{tender.id}] EXTRACTED CONTACTS: {extracted_contacts}")
                    else:
                        logger.info(f"[{source_table}:{tender.id}] NO CONTACTS EXTRACTED")
                    
                    # Log field counts
                    logger.info(f"[{source_table}:{tender.id}] FIELDS: {result.fields_before} → {result.fields_after} ({result.improvement_percentage:.1f}% improvement)")
                else:
                    logger.error(f"[{source_table}:{tender.id}] NORMALIZATION FAILED: {result.error}")
            
            # Add to results
            results_by_source[source_table] = source_results
            
            # Summarize source results
            successful = [r for r in source_results if r.success]
            logger.info(f"==== {source_table} SUMMARY: {len(successful)}/{len(source_results)} successful ====")
        
        return results_by_source
    
    def _get_url_fields_for_source(self, source_table: str) -> List[str]:
        """Get URL field names for a specific source table."""
        # Base URL fields to check in all sources
        base_fields = ["url", "link", "web_link"]
        
        # Source-specific URL fields
        source_fields = {
            "sam_gov": ["solicitation_link", "opportunity_link"],
            "wb": ["link"],
            "adb": ["link"],
            "ted_eu": ["notice_url"],
            "ungm": ["link"],
            "afd_tenders": ["notice_url"],
            "iadb": ["link"],
            "afdb": ["tender_url"],
            "aiib": ["link"]
        }
        
        # Combine base fields with source-specific fields
        return base_fields + source_fields.get(source_table, [])

    def _map_standard_tender_type(self, normalized_data: Dict[str, Any], tender: RawTender) -> None:
        """
        Map source-specific tender types to standard types accepted by our schema.
        
        Args:
            normalized_data: The normalized data dictionary to update
            tender: The raw tender with the original data
        """
        # Get the current tender type value
        tender_type = normalized_data.get("tender_type")
        
        # Standard mapping dictionary for known tender types
        standard_type_mapping = {
            # World Bank specific mappings
            "Request for Expression of Interest": "consulting",
            "Expression Of Interest": "consulting",
            "Contract Award": "other",
            "Invitation for Bids": "goods",
            "General Procurement Notice": "other",
            "Shortlist": "services",
            "Request for Proposal": "services",
            "Request for Bids": "goods",
            "Request for Quotations": "goods",
            "Procurement Plan": "other",
            "Prequalification": "other",
            
            # Generic mappings that might apply to multiple sources
            "Supplies": "goods",
            "Services": "services",
            "Works": "works",
            "Consulting": "consulting",
            "Consultancy": "consulting",
            "Mixed": "mixed",
            "Other": "other",
        }
        
        # If we have a tender type, try to map it
        if tender_type:
            # Try direct mapping
            if tender_type.lower() in [t.lower() for t in standard_type_mapping]:
                # Case-insensitive match
                for k, v in standard_type_mapping.items():
                    if k.lower() == tender_type.lower():
                        normalized_data["tender_type"] = v
                        return
            
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


# Create a singleton instance
normalizer = TenderNormalizer() 