"""
Main entry point for the tender normalizer.
"""
import asyncio
import json
import logging
import os
import sys
from typing import Dict, List, Optional, Any

import logfire

# Don't need to load environment variables with dotenv since Apify handles this
# from dotenv import load_dotenv

from .config import settings
from .models.tender import NormalizationResult, NormalizedTender, RawTender
from .services.db import supabase
from .services.normalizer import normalizer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)


def setup_logfire() -> None:
    """Set up monitoring with Logfire if a token is provided."""
    if settings.logfire_token:
        try:
            logfire.configure(
                token=settings.logfire_token.get_secret_value(),
                service_name="tender-normalizer-py",
                service_version="0.1.0",
            )
            
            # Instrument libraries
            logfire.instrument_httpx()
            logfire.instrument_asyncpg()
            logfire.instrument_pydantic()
            
            logger.info("Logfire monitoring configured")
        except Exception as e:
            logger.warning(f"Failed to configure Logfire: {str(e)}")
    else:
        logger.info("Logfire token not provided, monitoring disabled")


async def process_tenders(tenders: List[RawTender]) -> List[NormalizationResult]:
    """
    Process a list of tenders.
    
    Args:
        tenders: List of raw tenders to process
        
    Returns:
        List of normalization results
    """
    if not tenders:
        logger.info("No tenders to process")
        return []
    
    logger.info(f"Processing {len(tenders)} tenders")
    
    # Normalize tenders
    results = await normalizer.normalize_tenders(tenders)
    
    # Save successful normalizations to database
    successful = [r for r in results if r.success and r.normalized_tender]
    failed = [r for r in results if not r.success]
    
    logger.info(f"Normalization complete: {len(successful)} succeeded, {len(failed)} failed")
    
    # Save to database
    for result in successful:
        if result.normalized_tender:
            try:
                saved = await supabase.save_normalized_tender(result.normalized_tender)
                if not saved:
                    logger.error(f"Failed to save tender {result.tender_id} to database")
            except Exception as e:
                logger.error(f"Error saving tender {result.tender_id}: {str(e)}")
    
    # Print performance stats
    normalizer.log_performance_stats()
    
    return results


async def process_source(source_table: str, limit: int = 100) -> List[NormalizationResult]:
    """
    Process tenders from a specific source.
    
    Args:
        source_table: Name of the source table
        limit: Maximum number of tenders to process
        
    Returns:
        List of normalization results
    """
    logger.info(f"Processing tenders from {source_table}")
    
    # Get unprocessed tenders
    tenders = await supabase.get_unprocessed_tenders(source_table, limit)
    logger.info(f"Found {len(tenders)} unprocessed tenders in {source_table}")
    
    # Process tenders
    results = await process_tenders(tenders)
    
    return results


async def process_all_sources(
    limit_per_source: int = 25
) -> Dict[str, List[NormalizationResult]]:
    """
    Process tenders from all available sources.
    
    Args:
        limit_per_source: Maximum number of tenders to process per source
        
    Returns:
        Dictionary mapping source tables to normalization results
    """
    logger.info("Processing tenders from all sources")
    
    # Get tenders from all sources
    all_tenders = await supabase.get_unprocessed_tenders_from_all_sources(limit_per_source)
    logger.info(f"Found {len(all_tenders)} unprocessed tenders across all sources")
    
    # Group tenders by source for statistics
    tenders_by_source = {}
    for tender in all_tenders:
        if tender.source_table not in tenders_by_source:
            tenders_by_source[tender.source_table] = []
        tenders_by_source[tender.source_table].append(tender)
    
    # Log counts by source
    for source, source_tenders in tenders_by_source.items():
        logger.info(f"  {source}: {len(source_tenders)} tenders")
    
    # Process all tenders together
    results = await process_tenders(all_tenders)
    
    # Group results by source for return value
    results_by_source = {}
    for result in results:
        if result.source_table not in results_by_source:
            results_by_source[result.source_table] = []
        results_by_source[result.source_table].append(result)
    
    return results_by_source


async def process_test_batch(limit_per_source: int = 3) -> Dict[str, List[Dict[str, Any]]]:
    """
    Process a test batch of 3 tenders from each source table with extensive logging.
    
    This function is used to test and improve the normalization process by
    processing a small number of tenders from each source table and analyzing
    the results with detailed logging.
    
    Args:
        limit_per_source: Number of tenders to process from each source (default: 3)
        
    Returns:
        Dictionary mapping source tables to normalization results
    """
    logger.info(f"Processing test batch of {limit_per_source} tenders from each source")
    
    # Source tables to process
    source_tables = [
        "sam_gov", "wb", "adb", "ted_eu", "ungm", 
        "afd_tenders", "iadb", "afdb", "aiib"
    ]
    
    # Get tenders for each source
    tenders_by_source = {}
    total_tenders = 0
    
    for source in source_tables:
        try:
            tenders = await supabase.get_unprocessed_tenders(source, limit_per_source)
            if tenders:
                tenders_by_source[source] = tenders
                total_tenders += len(tenders)
                logger.info(f"Found {len(tenders)} tenders for test batch from {source}")
            else:
                logger.warning(f"No unprocessed tenders found for test batch from {source}")
        except Exception as e:
            logger.error(f"Error retrieving tenders from {source}: {str(e)}")
    
    if not total_tenders:
        logger.warning("No tenders found for test batch")
        return {}
    
    logger.info(f"Processing test batch with total of {total_tenders} tenders from {len(tenders_by_source)} sources")
    
    # Process test batch with special logging
    results = await normalizer.normalize_test_batch(tenders_by_source)
    
    # Mark tenders as processed
    for source, source_results in results.items():
        successful = [r for r in source_results if r.get('success') and r.get('normalized_data')]
        
        for result in successful:
            if result.get('normalized_data'):
                try:
                    # Create NormalizedTender from the normalized data
                    normalized_tender = NormalizedTender(
                        **result['normalized_data'],
                        normalized_method=result.get('method', 'direct_parsing'),
                        processing_time_ms=int(result.get('processing_time', 0) * 1000)
                    )
                    
                    saved = await supabase.save_normalized_tender(normalized_tender)
                    if not saved:
                        logger.error(f"Failed to save test tender {result['tender_id']} to database")
                except Exception as e:
                    logger.error(f"Error saving test tender {result['tender_id']}: {str(e)}")
    
    # Print performance stats
    normalizer.log_performance_stats()
    
    return results


def process_all_tenders(
    limit_per_source: int = 25,
    source_name: Optional[str] = None,
    test_mode: bool = False,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Synchronous wrapper for processing tenders.
    
    Args:
        limit_per_source: Maximum number of tenders to process per source
        source_name: Optional source table name to process just one source
        test_mode: Whether to run in test mode with extensive logging
        
    Returns:
        Dictionary mapping source tables to normalization results
    """
    if test_mode:
        # Process a small test batch with extensive logging
        return asyncio.run(process_test_batch(limit_per_source))
    elif source_name:
        # Process only the specified source
        return {source_name: asyncio.run(process_source(source_name, limit_per_source))}
    else:
        # Process all sources
        return asyncio.run(process_all_sources(limit_per_source))


def get_apify_input() -> dict:
    """Get input parameters from Apify."""
    apify_input = {}
    
    # Check if we're running on Apify
    if os.environ.get("APIFY_IS_AT_HOME"):
        try:
            # Try new Apify SDK first
            try:
                from apify import Actor
                logger.info("Using Apify SDK to get input")
                asyncio.run(Actor.init())
                apify_input = Actor.get_input() or {}
                logger.info(f"Loaded Apify input via SDK: {apify_input}")
            except (ImportError, Exception) as e:
                logger.warning(f"Failed to load input via Apify SDK: {str(e)}")
                
                # Fall back to reading the input file directly
                input_path = os.environ.get("APIFY_INPUT_KEY", "INPUT")
                if os.path.exists(input_path):
                    with open(input_path, encoding="utf-8") as f:
                        apify_input = json.load(f)
                        logger.info(f"Loaded Apify input from file: {apify_input}")
                else:
                    # Try loading from default locations
                    for possible_path in ["INPUT", "/input/INPUT", "./input/INPUT", "./INPUT"]:
                        if os.path.exists(possible_path):
                            with open(possible_path, encoding="utf-8") as f:
                                apify_input = json.load(f)
                                logger.info(f"Loaded Apify input from {possible_path}: {apify_input}")
                                break
        except Exception as e:
            logger.warning(f"Failed to load Apify input: {str(e)}")
            # Set default test mode for Apify to ensure we don't process too many tenders
            logger.info("Defaulting to test mode with 3 tenders per source")
            apify_input = {"testMode": True, "limit": 3}
    
    return apify_input


def main() -> None:
    """Main entry point."""
    # No need to load environment variables from .env
    # load_dotenv()
    
    # Set up Logfire monitoring
    setup_logfire()
    
    logger.info("Starting tender normalizer")
    
    # Get input parameters from Apify
    apify_input = get_apify_input()
    source_name = apify_input.get("sourceName")
    limit = int(apify_input.get("limit", settings.batch_size))
    
    # Default to test mode unless explicitly set to False
    test_mode = apify_input.get("testMode", True)
    
    # Process tenders
    if test_mode:
        # Use a smaller default limit for test mode if not explicitly set
        if "limit" not in apify_input and limit == settings.batch_size:
            limit = 3
        logger.info(f"Running in TEST MODE with {limit} tenders per source")
        results = process_all_tenders(limit, None, test_mode=True)
    elif source_name:
        logger.info(f"Processing source: {source_name} with limit: {limit}")
        results = process_all_tenders(limit, source_name)
    else:
        logger.info(f"Processing all sources with limit per source: {limit}")
        results = process_all_tenders(limit)
    
    # Print summary
    total_processed = sum(len(source_results) for source_results in results.values())
    logger.info(f"Processed {total_processed} tenders from {len(results)} sources")
    
    logger.info("Tender normalization complete")


if __name__ == "__main__":
    main() 