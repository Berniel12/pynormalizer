"""
Tender normalization service using PydanticAI.
"""
import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, TypeVar, Union

from langdetect import detect
from pydantic import BaseModel, Field, ValidationError
from pydantic_ai import Agent, RunContext
from pydantic_ai.exceptions import AgentError

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
        None, description="Notes about the normalization process"
    )


class TenderNormalizer:
    """Service for normalizing tender data using PydanticAI."""

    def __init__(self) -> None:
        """Initialize the tender normalizer."""
        # Set up the agent for normalization
        self.agent = Agent(
            settings.openai_model,
            result_type=NormalizationOutput,
            system_prompt=self._get_system_prompt(),
            instrument=bool(settings.logfire_token),
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
        """Get the system prompt for the normalization agent."""
        return """
        You are a specialized data normalization assistant for tender data. Your job is to transform
        raw tender data from various sources into a standardized format.
        
        Key principles to follow:
        1. Always prioritize accuracy and completeness of information.
        2. For critical fields (title, description, country), ensure they are never empty.
        3. If a field is missing in the input but can be inferred from other fields, do so.
        4. Make sure dates are in ISO format (YYYY-MM-DD).
        5. If a field is in a non-English language and there's no English version, translate it.
        6. Extract any embedded information that belongs in separate fields.
        7. Validate and correct country names to use full names (e.g., "USA" → "United States").
        8. Determine the appropriate tender status based on dates if status is missing.
        9. Extract and normalize financial information, separating amount from currency.
        10. Ensure organization names are properly formatted and include English versions when possible.
        
        The normalized tender data should be comprehensive, accurate, and follow the standardized schema.
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
        Normalize a tender using the LLM agent.
        
        Args:
            tender: The raw tender to normalize
            
        Returns:
            Normalization result
        """
        start_time = time.time()
        tender_dict = tender.model_dump(exclude={"source_data"})
        fields_before = self._count_non_empty_fields(tender_dict)
        
        try:
            # Run the agent with a timeout
            normalization_input = NormalizationInput(
                raw_tender=tender_dict,
                source_table=tender.source_table,
            )
            
            context = RunContext(timeout=settings.llm_timeout_seconds)
            result = await self.agent.run(normalization_input, context=context)
            
            # Create normalized tender model
            normalized_data = result.tender
            normalized_data["id"] = tender.id
            normalized_data["source_table"] = tender.source_table
            normalized_data["normalized_by"] = "pydantic-llm"
            normalized_data["normalized_at"] = datetime.utcnow().isoformat()
            
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
                method_used="llm",
                fields_before=fields_before,
                fields_after=fields_after,
                improvement_percentage=improvement,
            )
            
        except (AgentError, ValidationError, Exception) as e:
            # Fall back to rule-based normalization
            logger.warning(
                f"LLM normalization failed for {tender.id} from {tender.source_table}: {str(e)}"
            )
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
                "normalized_by": "rule-based-fallback",
                "normalized_at": datetime.utcnow().isoformat(),
            }
            
            # Map direct fields
            direct_field_mappings = [
                "title", "description", "country", "country_code", "location",
                "organization_name", "organization_id", "title_english",
                "description_english", "organization_name_english", "language",
            ]
            
            for field in direct_field_mappings:
                value = getattr(tender, field, None)
                if value and str(value).strip():
                    normalized_data[field] = value
            
            # Handle special fields
            # URL
            for url_field in ["url", "link", "web_link"]:
                url_value = getattr(tender, url_field, None)
                if url_value and str(url_value).strip():
                    normalized_data["url"] = url_value
                    break
            
            # Dates
            for date_field in ["publication_date", "deadline_date"]:
                date_value = getattr(tender, date_field, None)
                if date_value:
                    if isinstance(date_value, (datetime, date)):
                        normalized_data[date_field] = date_value
                    elif isinstance(date_value, str):
                        try:
                            # Simple ISO format parsing
                            normalized_data[date_field] = datetime.fromisoformat(
                                date_value.replace("Z", "+00:00")
                            )
                        except (ValueError, TypeError):
                            # Keep as is if parsing fails
                            pass
            
            # Status - convert string to enum
            status_value = getattr(tender, "status", None)
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
            if type_value:
                try:
                    normalized_data["tender_type"] = TenderType(type_value.lower())
                except (ValueError, TypeError):
                    normalized_data["tender_type"] = TenderType.UNKNOWN
            
            # Value and currency
            value = getattr(tender, "value", None)
            currency = getattr(tender, "currency", None)
            
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
            
            # Generate missing critical fields
            self._ensure_critical_fields(normalized_data, tender)
            
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
                improvement_percentage=0.0,
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
                    "ted": "European Union",
                    "un_tenders": "Global",
                }
                normalized_data["country"] = source_country_map.get(
                    tender.source_table, "Unknown"
                )

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