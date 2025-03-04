"""
Database service for connecting to Supabase.
"""
import logging
from typing import Any, Dict, List, Optional
import uuid
from datetime import datetime

import asyncpg
from supabase import Client, create_client
from pydantic import ValidationError

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
                
                # Map different ID fields based on the source table
                if source_table == "sam_gov" and "opportunity_id" in tender:
                    tender["id"] = str(tender["opportunity_id"])
                elif source_table == "iadb" and "project_number" in tender:
                    tender["id"] = str(tender["project_number"])
                elif "id" in tender:
                    # Convert numeric IDs to strings
                    tender["id"] = str(tender["id"])
                else:
                    # If no ID field could be found, generate one
                    logger.warning(f"No ID field found for {source_table} tender, generating UUID")
                    tender["id"] = f"{source_table}-{uuid.uuid4()}"
            
            # Convert to Pydantic models
            try:
                return [RawTender.model_validate(tender) for tender in response.data]
            except ValidationError as e:
                logger.error(f"Validation error for {source_table}: {str(e)}")
                return []
        
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
        Save a normalized tender to the unified_tenders table.
        
        Args:
            tender: The normalized tender to save
            
        Returns:
            True if saved successfully, False otherwise
        """
        client = self.connect()
        
        try:
            # Convert Pydantic model to dict
            tender_dict = tender.model_dump(mode="json")
            
            # Map fields to unified_tenders schema
            unified_tender = {
                # Required fields
                "title": tender_dict.get("title"),
                "source_table": tender_dict.get("source_table"),
                "source_id": tender_dict.get("source_id") or tender_dict.get("id"),
                
                # Optional fields with direct mapping
                "description": tender_dict.get("description"),
                "tender_type": tender_dict.get("tender_type"),
                "status": tender_dict.get("status"),
                "publication_date": tender_dict.get("publication_date"),
                "deadline_date": tender_dict.get("deadline_date"),
                "country": tender_dict.get("country"),
                "city": tender_dict.get("location"),  # Map location to city
                "organization_name": tender_dict.get("organization_name"),
                "organization_id": tender_dict.get("organization_id"),
                "url": tender_dict.get("url"),
                "language": tender_dict.get("language"),
                
                # English translation fields
                "title_english": tender_dict.get("title_english"),
                "description_english": tender_dict.get("description_english"),
                "organization_name_english": tender_dict.get("organization_name_english"),
                
                # Processing metadata
                "normalized_at": tender_dict.get("normalized_at"),
                "normalized_by": tender_dict.get("normalized_by"),
                "processed_at": datetime.utcnow().isoformat(),
                
                # Original data for reference
                "original_data": tender_dict.get("source_data"),
            }
            
            # Handle contact information if present
            if tender_dict.get("contact"):
                contact = tender_dict["contact"]
                unified_tender["contact_name"] = contact.get("name")
                unified_tender["contact_email"] = contact.get("email")
                unified_tender["contact_phone"] = contact.get("phone")
                unified_tender["contact_address"] = contact.get("address")
            
            # Handle document links if present
            if tender_dict.get("documents"):
                unified_tender["document_links"] = tender_dict["documents"]
            
            # Handle financial information
            if isinstance(tender_dict.get("value"), (int, float)):
                unified_tender["estimated_value"] = tender_dict["value"]
                unified_tender["currency"] = tender_dict.get("currency")
            
            # Insert into unified_tenders table
            response = client.table("unified_tenders").insert(unified_tender).execute()
            
            if response.data:
                # Update the original tender to mark it as processed
                update_response = None
                try:
                    # Different tables may have different primary key field names
                    if tender.source_table == "sam_gov" and hasattr(tender, "opportunity_id"):
                        # For sam_gov, the primary key is opportunity_id
                        update_response = (
                            client.table(tender.source_table)
                            .update({"processed": True})
                            .eq("opportunity_id", getattr(tender, "opportunity_id", tender.id))
                            .execute()
                        )
                    elif tender.source_table == "iadb" and hasattr(tender, "project_number"):
                        # For iadb, we can use project_number
                        update_response = (
                            client.table(tender.source_table)
                            .update({"processed": True})
                            .eq("project_number", getattr(tender, "project_number", tender.id))
                            .execute()
                        )
                    elif hasattr(tender, "source_id") and tender.source_id:
                        # Try using source_id as the primary key - this seems to be failing
                        # Let's modify this to use other fields
                        try:
                            # Map source tables to their known primary key fields based on error messages
                            table_pk_map = {
                                "sam_gov": "opportunity_id",
                                "wb": "notice_id",
                                "adb": "id",
                                "ted_eu": "id",
                                "ungm": "id",
                                "afd_tenders": "id",
                                "iadb": "project_number",
                                "afdb": "id",
                                "aiib": "id"
                            }
                            
                            pk_field = table_pk_map.get(tender.source_table, "id")
                            pk_value = tender.id
                            
                            # For sam_gov, we might need to get opportunity_id from original data
                            if tender.source_table == "sam_gov" and tender.source_data and "opportunity_id" in tender.source_data:
                                pk_value = tender.source_data["opportunity_id"]
                            
                            # Log what we're trying to use for debugging
                            logger.info(f"Updating processed flag for {tender.source_table} using {pk_field}={pk_value}")
                            
                            update_response = (
                                client.table(tender.source_table)
                                .update({"processed": True})
                                .eq(pk_field, pk_value)
                                .execute()
                            )
                        except Exception as e:
                            logger.error(f"Error updating processed flag using mapped PK for {tender.source_table}: {str(e)}")
                    else:
                        # Try some common primary key names
                        for pk_field in ["id", "tender_id", "_id", "notice_id", "record_id"]:
                            try:
                                update_response = (
                                    client.table(tender.source_table)
                                    .update({"processed": True})
                                    .eq(pk_field, tender.id)
                                    .execute()
                                )
                                break
                            except Exception:
                                continue
                        
                        # If all attempts failed, log a warning but don't fail the save
                        if not update_response:
                            logger.warning(
                                f"Could not update processed flag for {tender.source_table} tender {tender.id}. "
                                f"Consider manually marking it as processed."
                            )
                    
                    logger.info(f"Saved unified tender from {tender.source_table} with source ID {tender.id}")
                    return True
                except Exception as e:
                    logger.error(f"Error updating processed flag: {str(e)}")
                    # Still return True because the unified tender was saved successfully
                    return True
            
            return False
        
        except Exception as e:
            logger.error(f"Error saving unified tender {tender.id}: {str(e)}")
            return False


# Create a singleton instance
supabase = SupabaseClient() 