"""
MongoDB utilities for chatbot instances and conversation history management

Collections:
- instances: Chatbot instance metadata
- chat_history: Chat message history
- checkpoints: LangGraph conversation memory (managed by MongoDBSaver)
"""

from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, DuplicateKeyError
from datetime import datetime
from typing import Optional, Dict, List, Any
from utils.logger import log_info, log_error, log_exception
import os


class MongoDBManager:
    """
    MongoDB Manager for handling chatbot instances and conversation history
    """
    
    def __init__(self, mongodb_uri: str, database_name: str = "Inshoraa"):
        """
        Initialize MongoDB Manager
        
        Args:
            mongodb_uri: MongoDB connection URI
            database_name: Name of the database (default: "python")
        """
        try:
            self.client = MongoClient(mongodb_uri)
            # Test connection
            self.client.admin.command('ping')
            self.db = self.client[database_name]
            
         
            self.transcripts_collection = self.db["transcripts"]
            
            # Create indexes for better performance
            self._create_indexes()
            
            log_info(f"MongoDB Manager initialized successfully with database: {database_name}")
        except ConnectionFailure as e:
            log_error(f"Failed to connect to MongoDB: {str(e)}")
            raise
        except Exception as e:
            log_exception(f"Error initializing MongoDB Manager: {str(e)}")
            raise
    
    def _create_indexes(self):
        """Create indexes for collections"""
        try:
            # Indexes for transcripts collection
            self.transcripts_collection.create_index("caller_id")
            self.transcripts_collection.create_index("timestamp")
            self.transcripts_collection.create_index("contact_number")
            
            log_info("MongoDB indexes created successfully")
        except Exception as e:
            log_error(f"Error creating indexes: {str(e)}")
    
    # ==================== Transcript Methods ====================
    
    def save_transcript(
        self,
        transcript: Dict[str, Any],
        caller_id: str,
        name: str,
        contact_number: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Save a call transcript to the transcripts collection
        
        Args:
            transcript: Transcript data (usually from session.history.to_dict())
            caller_id: Unique identifier for the caller
            name: Caller's name
            contact_number: Caller's contact number (optional)
            metadata: Additional metadata (optional)
            
        Returns:
            Transcript document ID
        """
        try:
            transcript_data = {
                "transcript": transcript,
                "caller_id": caller_id,
                "name": name,
                "contact_number": contact_number,
                "timestamp": datetime.utcnow(),
                "metadata": metadata or {}
            }
            
            result = self.transcripts_collection.insert_one(transcript_data)
            transcript_id = str(result.inserted_id)
            
            log_info(f"Saved transcript for caller: {name} (ID: {caller_id})")
            return transcript_id
            
        except Exception as e:
            log_exception(f"Error saving transcript: {str(e)}")
            raise
    
    def get_transcript(self, caller_id: str) -> Optional[Dict[str, Any]]:
        """
        Get transcript by caller ID
        
        Args:
            caller_id: Caller identifier
            
        Returns:
            Transcript data or None if not found
        """
        try:
            transcript = self.transcripts_collection.find_one({"caller_id": caller_id})
            if transcript:
                transcript["_id"] = str(transcript["_id"])
                log_info(f"Retrieved transcript for caller ID: {caller_id}")
            return transcript
        except Exception as e:
            log_exception(f"Error retrieving transcript: {str(e)}")
            return None
    
    def get_transcripts_by_contact_number(
        self,
        contact_number: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get transcripts by contact number
        
        Args:
            contact_number: Contact number to search for
            limit: Maximum number of transcripts to return
            
        Returns:
            List of transcript data
        """
        try:
            transcripts = list(
                self.transcripts_collection
                .find({"contact_number": contact_number})
                .sort("timestamp", -1)
                .limit(limit)
            )
            
            for transcript in transcripts:
                transcript["_id"] = str(transcript["_id"])
            
            log_info(f"Retrieved {len(transcripts)} transcripts for contact: {contact_number}")
            return transcripts
        except Exception as e:
            log_exception(f"Error retrieving transcripts by contact number: {str(e)}")
            return []
    
    # ==================== Connection Management ====================
    
    def close(self):
        """Close MongoDB connection"""
        try:
            self.client.close()
            log_info("MongoDB connection closed")
        except Exception as e:
            log_error(f"Error closing MongoDB connection: {str(e)}")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()


# Singleton instance
_mongodb_manager = None


def get_mongodb_manager(mongodb_uri: Optional[str] = None, database_name: str = "Inshoraa") -> MongoDBManager:
    """
    Get or create MongoDB Manager singleton instance
    
    Args:
        mongodb_uri: MongoDB connection URI (required for first call)
        database_name: Database name (default: "python")
        
    Returns:
        MongoDBManager instance
    """
    global _mongodb_manager
    
    if _mongodb_manager is None:
        if mongodb_uri is None:
            # Try to get from environment
            mongodb_uri = os.getenv("MONGODB_URI")
            if mongodb_uri is None:
                raise ValueError("MongoDB URI must be provided or set in MONGODB_URI environment variable")
        
        _mongodb_manager = MongoDBManager(mongodb_uri, database_name)
    
    return _mongodb_manager
