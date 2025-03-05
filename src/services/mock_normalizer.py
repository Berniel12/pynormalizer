import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union

from ..config import NormalizerConfig

logger = logging.getLogger(__name__)

class MockNormalizer:
    """
    A mock normalizer that simulates the normalization process without making actual API calls.
    This is used for testing purposes when a valid API key is not available.
    """

    def __init__(
        self,
        api_key: str = "mock-api-key",
        model: str = "gpt-4o-mini",
        config: Optional[NormalizerConfig] = None,
    ):
        """
        Initialize the MockNormalizer.
        
        Args:
            api_key: OpenAI API key (not used in mock)
            model: OpenAI model to use (not used in mock)
            config: Normalizer configuration
        """
        self.api_key = api_key
        self.model = model
        self.config = config or NormalizerConfig()
        logger.info(f"Initialized MockNormalizer with model: {model}")

    async def normalize_tender(self, tender: Dict[str, Any]) -> Dict[str, Any]:
        """
        Mock normalize a tender without making API calls.
        
        Args:
            tender: The raw tender data to normalize
            
        Returns:
            A dictionary containing the normalized data and metadata
        """
        tender_id = tender.get("id", "unknown")
        source = tender.get("source_table", "unknown")
        
        logger.info(f"Mock normalizing tender {tender_id} from {source}")
        
        # Simulate processing time
        time.sleep(0.5)
        
        # Create a mock normalized response
        normalized_data = {
            "title": tender.get("title", "Unknown Title"),
            "source_table": source,
            "source_id": tender_id,
            "description": tender.get("description", ""),
            "tender_type": "unknown",
            "status": "active",
            "publication_date": tender.get("publication_date", "").split("T")[0] if tender.get("publication_date") else None,
            "country": tender.get("country", "Unknown"),
            "organization_name": tender.get("organization_name", "Unknown Organization"),
            "url": tender.get("url", "")
        }
        
        # Simulate missing fields
        missing_fields = []
        for field in ["deadline_date", "city", "organization_id", "buyer", "project_name"]:
            if not tender.get(field):
                missing_fields.append(field)
        
        # Create the response
        response = {
            "normalized_data": normalized_data,
            "used_llm": False,
            "method": "mock",
            "processing_time": 0.5,
            "error": None,
            "missing_fields": missing_fields,
            "notes": "This is a mock normalization for testing purposes."
        }
        
        return response 