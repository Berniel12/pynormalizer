# Tender Normalizer for Apify

This is a Python-based implementation of the tender normalization system using Pydantic and PydanticAI, configured for deployment on Apify.

## Overview

The Tender Normalizer processes tenders from various sources stored in Supabase, normalizes them using a combination of LLM (GPT-4o-mini) and rule-based approaches, and saves the normalized tenders back to the database.

## Apify Deployment

### Environment Variables

The following environment variables should be configured in your Apify actor:

- `SUPABASE_URL`: URL of your Supabase instance
- `SUPABASE_KEY`: Supabase API key
- `OPENAI_API_KEY`: OpenAI API key for GPT-4o-mini
- `LOGFIRE_TOKEN` (optional): Token for Logfire monitoring
- `USE_LLM_NORMALIZATION`: Set to "true" or "false" to enable/disable LLM (default: "true")
- `LLM_TIMEOUT_SECONDS`: Timeout in seconds for LLM normalization (default: "60")
- `BATCH_SIZE`: Number of tenders to process in each batch (default: "25")
- `CONCURRENT_REQUESTS`: Number of concurrent normalization requests (default: "5")

### Deploying to Apify

1. Create a new actor on the [Apify platform](https://console.apify.com/)
2. Upload this directory as a ZIP file or connect to your Git repository
3. Configure the above environment variables
4. Deploy the actor

### Running the Actor

The actor runs in test mode by default:
1. By default, it processes 3 tenders per source with extensive analysis logging
2. To process more tenders, set the `limit` parameter
3. To disable test mode and process all tenders in the normal mode, set `testMode` to `false`

In test mode, the actor will:
1. Connect to Supabase
2. Retrieve a small number of unprocessed tenders from each source
3. Normalize them with detailed logging of the process
4. Save the normalized tenders back to Supabase
5. Provide comprehensive analysis logs in the key-value store

## Analysis Logs

When running in test mode, detailed analysis logs are saved to:
- The Apify key-value store as `normalization_analysis.log`
- Standard output (visible in the Apify console)

These logs contain:
- Field comparison (raw vs. normalized)
- URL extraction analysis
- Contact information tracking
- Success/failure statistics
- Error patterns and suggestions
- Field extraction success rates

## File Structure

- `src/`: Source code
  - `models/`: Pydantic data models
  - `services/`: Business logic services
  - `config.py`: Configuration settings
  - `main.py`: Application entry point
- `Dockerfile`: Docker configuration for Apify
- `apify.json`: Apify actor configuration
- `pyproject.toml`: Python project dependencies

## Monitoring

If a `LOGFIRE_TOKEN` is provided, the actor will automatically configure Logfire monitoring, providing insights into:

- Normalization performance
- LLM usage and timeouts
- Database operations
- Field improvement rates
- Success/failure statistics 