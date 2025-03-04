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

The actor will automatically:
1. Connect to Supabase
2. Retrieve unprocessed tenders from all sources
3. Normalize them using LLM or rule-based methods as appropriate
4. Save the normalized tenders back to Supabase
5. Log performance statistics

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