"""
Database service for connecting to Supabase.
"""
import logging
from typing import Any, Dict, List, Optional

import asyncpg
from supabase import Client, create_client

from ..config import settings
from ..models.tender import NormalizedTender, RawTender

logger = logging.getLogger(__name__)


class SupabaseClient:
    """Client for interacting with Supabase database."""

    def __init__(self) -> None:
        """Initialize the Supabase client."""
        self.client: Optional[Client] = None
        self.initialized = False
        
    def connect(self) -> Client:
        """Connect to Supabase and return the client."""
        if not self.initialized:
            if not settings.supabase_url or not settings.supabase_key.get_secret_value():
                raise ValueError("Supabase URL and key must be provided")
            
            self.client = create_client(
                settings.supabase_url, settings.supabase_key.get_secret_value()
            )
            self.initialized = True
            logger.info("Connected to Supabase")
        
        return self.client

    async def get_unprocessed_tenders(
        self, source_table: str, limit: int = 100
    ) -> List[RawTender]:
        """
        Get unprocessed tenders from a specific source table.
        
        Args:
            source_table: Name of the source table
            limit: Maximum number of tenders to retrieve
            
        Returns:
            List of unprocessed tenders
        """
        client = self.connect()
        
        # Query for unprocessed tenders
        response = client.table(source_table).select("*").eq("processed", False).limit(limit).execute()
        
        if response.data:
            # Add source_table to each tender
            for tender in response.data:
                tender["source_table"] = source_table
            
            # Convert to Pydantic models
            return [RawTender.model_validate(tender) for tender in response.data]
        
        return []

    async def get_unprocessed_tenders_from_all_sources(
        self, limit_per_source: int = 25
    ) -> List[RawTender]:
        """
        Get unprocessed tenders from all source tables.
        
        Args:
            limit_per_source: Maximum number of tenders to retrieve per source
            
        Returns:
            List of unprocessed tenders from all sources
        """
        source_tables = [
            "sam_gov", "wb", "adb", "ted_eu", "ungm", 
            "afd_tenders", "iadb", "afdb", "aiib"
        ]
        
        all_tenders = []
        for source in source_tables:
            try:
                tenders = await self.get_unprocessed_tenders(source, limit_per_source)
                all_tenders.extend(tenders)
                logger.info(f"Retrieved {len(tenders)} unprocessed tenders from {source}")
            except Exception as e:
                logger.error(f"Error retrieving tenders from {source}: {str(e)}")
        
        return all_tenders

    async def save_normalized_tender(self, tender: NormalizedTender) -> bool:
        """
        Save a normalized tender to the normalized_tenders table.
        
        Args:
            tender: The normalized tender to save
            
        Returns:
            True if saved successfully, False otherwise
        """
        client = self.connect()
        
        try:
            # Convert Pydantic model to dict
            tender_dict = tender.model_dump(mode="json")
            
            # Insert into normalized_tenders table
            response = client.table("normalized_tenders").insert(tender_dict).execute()
            
            if response.data:
                # Update the original tender to mark it as processed
                update_response = (
                    client.table(tender.source_table)
                    .update({"processed": True, "normalized": True})
                    .eq("id", tender.id)
                    .execute()
                )
                
                logger.info(f"Saved normalized tender {tender.id} from {tender.source_table}")
                return True
            
            return False
        
        except Exception as e:
            logger.error(f"Error saving normalized tender {tender.id}: {str(e)}")
            return False


# Create a singleton instance
supabase = SupabaseClient() 