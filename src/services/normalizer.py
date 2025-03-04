"""
Tender normalization service using PydanticAI.
"""
import asyncio
import json
import logging
import os
import time
from datetime import datetime, date
from typing import Any, Dict, List, Optional, TypeVar, Union

from langdetect import detect
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

T = TypeVar("T", bound=BaseModel)


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
            if field in tender and tender[field] and not isinstance(tender[field], (str, datetime, date)):
                raise ValueError(f"Invalid type for {field}: must be a date/datetime or ISO string")
        
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
        self.agent = Agent(
            settings.openai_model,
            result_type=NormalizationOutput,
            system_prompt=self._get_system_prompt(),
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
        - title: Short, descriptive title of the tender opportunity
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
            context = RunContext(model="mistral", temperature=0.1)
            
            # Initialize the agent
            try:
                agent = Agent(
                    task="Normalize tender data",
                    description=self._get_system_prompt(),
                )
            except Exception as e:
                logger.error(f"Failed to initialize agent: {str(e)}")
                # Try without optional parameters if we get an error
                agent = Agent(description=self._get_system_prompt())
                
            # Run the normalization
            result = await agent.run(input_data, context=context)
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
            
            # Handle dates
            self._infer_status_from_dates(tender_data)
            
            # Ensure all critical fields are present
            self._ensure_critical_fields(tender_data, tender)
            
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
            logger.error(f"LLM normalization failed for {tender.id}: {str(e)}")
            logger.info(f"Falling back to rule-based normalization for {tender.id}")
            
            # Use the fallback method
            return await self._normalize_with_fallback(
                tender, error=str(e), start_time=start_time
            )

    async def _normalize_with_fallback(
        self, tender: RawTender, error: str = "Fallback used", start_time: Optional[float] = None
    ) -> NormalizationResult:
        """
        Normalize a tender using rule-based fallback methods.
        
        Args:
            tender: The raw tender to normalize
            error: Error message that triggered fallback
            start_time: Start time for processing (if already started)
            
        Returns:
            Normalization result
        """
        if start_time is None:
            start_time = time.time()
            
        tender_dict = tender.model_dump(exclude={"source_data"})
        fields_before = self._count_non_empty_fields(tender_dict)
        
        try:
            # Copy basic fields
            normalized_data = {
                "id": tender.id,
                "source_table": tender.source_table,
                "source_id": getattr(tender, "source_id", tender.id),
                "normalized_by": "rule-based-fallback",
                "normalized_method": "fallback",
                "fallback_reason": error,
                "normalized_at": datetime.utcnow().isoformat(),
            }
            
            # Map direct fields
            direct_field_mappings = [
                "title", "description", "country", "country_code", "location",
                "organization_name", "organization_id", "title_english",
                "description_english", "organization_name_english", "language",
                "buyer", "project_name", "project_id", "project_number", "sector",
                "reference_number", "notice_id", "procurement_method",
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
                }
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
                        
                        if value and str(value).strip():
                            normalized_data[norm_field] = value
            
            # Then apply direct mappings for any fields not already mapped
            for field in direct_field_mappings:
                if field not in normalized_data:
                    # Try to get from attributes first
                    value = getattr(tender, field, None)
                    # If not found, try to get from source_data
                    if value is None and hasattr(tender, "source_data") and tender.source_data:
                        value = tender.source_data.get(field)
                    
                    if value and str(value).strip():
                        normalized_data[field] = value
            
            # Handle URL fields
            for url_field in ["url", "link", "web_link"]:
                url_value = getattr(tender, url_field, None)
                if url_value is None and hasattr(tender, "source_data") and tender.source_data:
                    url_value = tender.source_data.get(url_field)
                
                if url_value and str(url_value).strip():
                    normalized_data["url"] = url_value
                    break
            
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
                normalized_data["documents"] = documents
            
            # Contact information
            contact_fields = {
                "name": ["contact_name", "point_of_contact", "contact_person"],
                "email": ["contact_email", "email", "point_of_contact_email"],
                "phone": ["contact_phone", "phone", "telephone", "point_of_contact_phone"],
                "address": ["contact_address", "address"]
            }
            
            contact_info = {}
            for field, sources in contact_fields.items():
                for source in sources:
                    value = getattr(tender, source, None)
                    if value is None and hasattr(tender, "source_data") and tender.source_data:
                        value = tender.source_data.get(source)
                    
                    if value and str(value).strip():
                        contact_info[field] = value
                        break
            
            if contact_info:
                normalized_data["contact"] = contact_info
            
            # Dates
            for date_field in ["publication_date", "deadline_date"]:
                date_value = getattr(tender, date_field, None)
                if date_value is None and hasattr(tender, "source_data") and tender.source_data:
                    date_value = tender.source_data.get(date_field)
                
                if date_value:
                    if isinstance(date_value, (datetime, date)):
                        normalized_data[date_field] = date_value
                    elif isinstance(date_value, str):
                        try:
                            # Try multiple date formats
                            for fmt in [
                                "%Y-%m-%dT%H:%M:%S", 
                                "%Y-%m-%d %H:%M:%S",
                                "%Y-%m-%d",
                                "%d/%m/%Y",
                                "%m/%d/%Y"
                            ]:
                                try:
                                    parsed_date = datetime.strptime(date_value, fmt)
                                    normalized_data[date_field] = parsed_date
                                    break
                                except ValueError:
                                    continue
                            
                            # If none of the formats worked, try ISO format
                            if date_field not in normalized_data:
                                normalized_data[date_field] = datetime.fromisoformat(
                                    date_value.replace("Z", "+00:00")
                                )
                        except (ValueError, TypeError):
                            # Keep as is if parsing fails
                            normalized_data[date_field] = date_value
            
            # Status - convert string to enum
            status_value = getattr(tender, "status", None)
            if status_value is None and hasattr(tender, "source_data") and tender.source_data:
                status_value = tender.source_data.get("status")
            
            if status_value:
                try:
                    normalized_data["status"] = TenderStatus(status_value.lower())
                except (ValueError, TypeError):
                    # Infer status from dates
                    self._infer_status_from_dates(normalized_data)
            else:
                # Infer status from dates
                self._infer_status_from_dates(normalized_data)
            
            # Tender type - convert string to enum
            type_value = getattr(tender, "tender_type", None)
            if type_value is None and hasattr(tender, "source_data") and tender.source_data:
                type_value = tender.source_data.get("tender_type")
            
            if type_value:
                try:
                    normalized_data["tender_type"] = TenderType(type_value.lower())
                except (ValueError, TypeError):
                    normalized_data["tender_type"] = TenderType.UNKNOWN
            
            # Value and currency
            value = getattr(tender, "value", None)
            if value is None and hasattr(tender, "source_data") and tender.source_data:
                value = tender.source_data.get("value")
            
            currency = getattr(tender, "currency", None)
            if currency is None and hasattr(tender, "source_data") and tender.source_data:
                currency = tender.source_data.get("currency")
            
            if value is not None:
                if isinstance(value, (int, float)):
                    normalized_data["value"] = float(value)
                elif isinstance(value, str):
                    try:
                        normalized_data["value"] = float(value.replace(",", ""))
                    except (ValueError, TypeError):
                        pass
                elif isinstance(value, dict) and "amount" in value:
                    normalized_data["value"] = float(value["amount"])
                    if "currency" in value and not currency:
                        normalized_data["currency"] = value["currency"]
            
            if currency:
                normalized_data["currency"] = currency
            
            # Include original data
            if hasattr(tender, "source_data") and tender.source_data:
                normalized_data["source_data"] = tender.source_data
            
            # Generate missing critical fields
            self._ensure_critical_fields(normalized_data, tender)
            
            # Calculate stats
            processing_time = time.time() - start_time
            processing_time_ms = int(processing_time * 1000)
            
            # Include processing time in milliseconds
            normalized_data["processing_time_ms"] = processing_time_ms
            
            # Create normalized tender model
            normalized_tender = NormalizedTender.model_validate(normalized_data)
            
            # Calculate stats
            processing_time = time.time() - start_time
            fields_after = self._count_non_empty_fields(normalized_tender.model_dump())
            improvement = (
                ((fields_after - fields_before) / fields_before) * 100
                if fields_before > 0
                else 0
            )
            
            return NormalizationResult(
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

    def _infer_status_from_dates(self, normalized_data: Dict[str, Any]) -> None:
        """Infer tender status from dates."""
        deadline = normalized_data.get("deadline_date")
        publication = normalized_data.get("publication_date")
        
        if not deadline:
            normalized_data["status"] = TenderStatus.UNKNOWN
            return
        
        now = datetime.utcnow()
        
        if isinstance(deadline, str):
            try:
                deadline = datetime.fromisoformat(deadline.replace("Z", "+00:00"))
            except (ValueError, TypeError):
                normalized_data["status"] = TenderStatus.UNKNOWN
                return
        
        if deadline > now:
            normalized_data["status"] = TenderStatus.ACTIVE
        else:
            normalized_data["status"] = TenderStatus.CLOSED

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


# Create a singleton instance
normalizer = TenderNormalizer() 