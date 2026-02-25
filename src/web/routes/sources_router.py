"""
sources routes - ingest data from external sources via HTTP

POST /sources/{source}/ingest
  body: { creds: {...}, filters: {...}, user_id?: string }

POST /sources/webhook/{source}
  generic webhook endpoint for source-specific payloads
"""
from fastapi import APIRouter
from typing import Optional, Dict, Any
from pydantic import BaseModel

router = APIRouter(prefix="/sources", tags=["sources"])

class ingest_req(BaseModel):
    creds: Dict[str, Any] = {}
    filters: Dict[str, Any] = {}
    user_id: Optional[str] = None