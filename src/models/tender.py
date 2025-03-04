"""
Pydantic models for tender data.
"""
from datetime import date, datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import (
    BaseModel,
    Field,
    HttpUrl,
    field_validator,
    model_validator,
)


class TenderStatus(str, Enum):
    """Possible tender status values."""
    
    ACTIVE = "active"
    CLOSED = "closed"
    AWARDED = "awarded"
    CANCELED = "canceled"
    UPCOMING = "upcoming"
    UNKNOWN = "unknown"


class TenderType(str, Enum):
    """Possible tender type values."""
    
    GOODS = "goods"
    SERVICES = "services"
    WORKS = "works"
    CONSULTING = "consulting"
    MIXED = "mixed"
    OTHER = "other"
    UNKNOWN = "unknown"


class MoneyAmount(BaseModel):
    """Model representing a monetary amount with currency."""
    
    amount: float = Field(..., description="The numeric amount")
    currency: str = Field(..., description="The currency code (e.g., USD, EUR)")


class ContactInfo(BaseModel):
    """Contact information for the tender."""
    
    name: Optional[str] = Field(None, description="Name of the contact person")
    email: Optional[str] = Field(None, description="Email address")
    phone: Optional[str] = Field(None, description="Phone number")
    address: Optional[str] = Field(None, description="Physical address")


class DocumentLink(BaseModel):
    """Link to a tender document."""
    
    title: Optional[str] = Field(None, description="Title of the document")
    url: HttpUrl = Field(..., description="URL to the document")
    description: Optional[str] = Field(None, description="Description of the document")
    language: Optional[str] = Field(None, description="Language of the document")


class RawTender(BaseModel):
    """
    Raw tender data as it exists in the database before normalization.
    This model is flexible to accommodate different source formats.
    """
    
    id: str = Field(..., description="Unique identifier for the tender")
    source_table: str = Field(..., description="Source table name (e.g., 'sam_gov', 'wb')")
    
    # Basic tender information - these fields might be named differently in different sources
    title: Optional[str] = None
    description: Optional[str] = None
    
    # Dates
    publication_date: Optional[Union[str, datetime, date]] = None
    deadline_date: Optional[Union[str, datetime, date]] = None
    
    # Location information
    country: Optional[str] = None
    country_code: Optional[str] = None
    location: Optional[str] = None
    
    # Organization
    organization_name: Optional[str] = None
    organization_id: Optional[str] = None
    
    # Multi-language fields
    title_english: Optional[str] = None
    description_english: Optional[str] = None
    organization_name_english: Optional[str] = None
    
    # Classification
    status: Optional[str] = None
    tender_type: Optional[str] = None
    
    # Financial information
    value: Optional[Union[float, dict, str]] = None
    currency: Optional[str] = None
    
    # Other metadata
    language: Optional[str] = None
    url: Optional[str] = None
    link: Optional[str] = None
    web_link: Optional[str] = None
    
    # These fields allow for any additional source-specific data
    source_id: Optional[str] = None
    source_data: Optional[Dict[str, Any]] = None
    
    # Processing metadata
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    normalized: Optional[bool] = Field(default=False)
    processed: Optional[bool] = Field(default=False)
    
    # Allow additional fields
    model_config = {
        "extra": "allow",
    }


class NormalizedTender(BaseModel):
    """
    Normalized tender data with consistent field names and formats.
    This is the model we aim to transform raw tenders into.
    """
    
    # Identifier and source information
    id: str = Field(..., description="Unique identifier for the tender")
    source_table: str = Field(..., description="Source table name")
    source_id: Optional[str] = Field(None, description="Original ID from the source")
    
    # Core tender information
    title: str = Field(..., description="Title of the tender")
    description: str = Field(..., description="Description of the tender")
    
    # Dates
    publication_date: Optional[datetime] = Field(None, description="Date of publication")
    deadline_date: Optional[datetime] = Field(None, description="Submission deadline")
    
    # Location information
    country: str = Field(..., description="Country where the tender is located")
    country_code: Optional[str] = Field(None, description="ISO country code")
    location: Optional[str] = Field(None, description="Specific location within the country")
    
    # Organization
    organization_name: Optional[str] = Field(None, description="Name of the issuing organization")
    organization_id: Optional[str] = Field(None, description="ID of the issuing organization")
    
    # Multi-language fields
    title_english: Optional[str] = Field(None, description="English version of the title")
    description_english: Optional[str] = Field(None, description="English version of the description")
    organization_name_english: Optional[str] = Field(
        None, description="English version of the organization name"
    )
    
    # Classification
    status: TenderStatus = Field(
        default=TenderStatus.UNKNOWN, description="Current status of the tender"
    )
    tender_type: TenderType = Field(
        default=TenderType.UNKNOWN, description="Type of the tender"
    )
    
    # Financial information
    value: Optional[float] = Field(None, description="Value of the tender")
    currency: Optional[str] = Field(None, description="Currency of the tender value")
    
    # URL and documents
    url: Optional[HttpUrl] = Field(None, description="Main URL of the tender")
    documents: Optional[List[DocumentLink]] = Field(
        default_factory=list, description="Associated documents"
    )
    
    # Contact information
    contact: Optional[ContactInfo] = Field(None, description="Contact information")
    
    # Language information
    language: Optional[str] = Field(None, description="Original language of the tender")
    
    # Normalization metadata
    normalized_at: datetime = Field(default_factory=datetime.utcnow)
    normalized_by: str = Field(default="pydantic-llm")
    
    # Validation to ensure critical fields are present
    @model_validator(mode="after")
    def check_critical_fields(self) -> "NormalizedTender":
        """Ensure all critical fields are present and valid."""
        # Check title
        if not self.title or self.title.strip() == "":
            raise ValueError("Title cannot be empty")
            
        # Check description
        if not self.description or self.description.strip() == "":
            raise ValueError("Description cannot be empty")
            
        # Check country
        if not self.country or self.country.strip() == "":
            raise ValueError("Country cannot be empty")
            
        return self
    
    # Field validators for specific fields
    @field_validator("country")
    @classmethod
    def validate_country(cls, v: Optional[str]) -> str:
        """Validate and normalize country names."""
        if not v:
            raise ValueError("Country cannot be empty")
        
        # Normalize common country name variations
        country_map = {
            "usa": "United States",
            "united states of america": "United States",
            "u.s.a": "United States",
            "u.s.": "United States",
            "uk": "United Kingdom",
            "england": "United Kingdom",
            "great britain": "United Kingdom",
        }
        
        normalized = v.lower().strip()
        if normalized in country_map:
            return country_map[normalized]
        
        return v.strip()


class NormalizationResult(BaseModel):
    """Result of a normalization operation."""
    
    tender_id: str = Field(..., description="ID of the processed tender")
    source_table: str = Field(..., description="Source table of the tender")
    success: bool = Field(..., description="Whether normalization was successful")
    normalized_tender: Optional[NormalizedTender] = Field(
        None, description="The normalized tender, if successful"
    )
    error: Optional[str] = Field(None, description="Error message, if unsuccessful")
    processing_time: float = Field(..., description="Time taken for processing in seconds")
    method_used: str = Field(..., description="Normalization method used (llm, fallback, etc.)")
    fields_before: int = Field(..., description="Number of fields before normalization")
    fields_after: int = Field(..., description="Number of fields after normalization")
    improvement_percentage: float = Field(
        ..., description="Percentage improvement in field completeness"
    ) 