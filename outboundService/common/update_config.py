"""
Dynamic configuration manager for voice agent service.

This module provides functionality to update and read dynamic configuration
parameters from a JSON file. This allows the agent to pick up new configuration
values for each call without requiring environment variable updates or restarts.

Thread-safe and async-safe implementation for production use.
"""

import json
import asyncio
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from threading import Lock

# Configure logger (using ASCII-safe log messages for Windows compatibility)
logger = logging.getLogger("update_config")

# Path to config file (in project root)
CONFIG_FILE = Path(__file__).parent.parent.parent / "config.json"

# Thread lock for file write safety
_write_lock = Lock()


def update_config(
    caller_name: Optional[str] = None,
    agent_instructions: Optional[str] = None,
    tts_language: str = "en",
    voice_id: str = "21m00Tcm4TlvDq8ikWAM",
    additional_params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Update the dynamic configuration file (config.json) with new parameters.
    
    This function is thread-safe and can be called from multiple threads or
    async contexts. It creates the config file if it doesn't exist.
    
    Args:
        caller_name: Name of the person being called (optional)
        agent_instructions: Custom instructions for the AI agent (optional)
        tts_language: TTS language code (e.g., "en", "es", "fr", "de")
        voice_id: ElevenLabs voice ID (default: "21m00Tcm4TlvDq8ikWAM" - Rachel)
        additional_params: Any additional parameters to store in config (optional)
    
    Returns:
        Dict containing the updated configuration
        
    Raises:
        IOError: If unable to write to config file
        json.JSONDecodeError: If existing config file has invalid JSON
    """
    try:
        # Build the configuration dictionary
        config_data = {
            "caller_name": caller_name or "Guest",
            "agent_instructions": agent_instructions or "You are a helpful voice AI assistant.",
            "tts_language": tts_language,
            "voice_id": voice_id,
            "last_updated": asyncio.get_event_loop().time() if asyncio._get_running_loop() else 0
        }
        
        # Add any additional parameters
        if additional_params:
            config_data.update(additional_params)
        
        # Thread-safe write to file
        with _write_lock:
            # Ensure the config file's parent directory exists
            CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
            
            # Write configuration to file with pretty formatting
            with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=4, ensure_ascii=False)
        
        logger.info(f"[OK] Configuration updated successfully")
        logger.info(f"  - Caller Name: {config_data['caller_name']}")
        logger.info(f"  - Agent Instructions: {config_data['agent_instructions'][:100]}...")
        logger.info(f"  - TTS Language: {config_data['tts_language']}")
        logger.info(f"  - Voice ID: {config_data['voice_id']}")
        
        # Log additional parameters if present
        if config_data.get('transfer_to'):
            logger.info(f"  - Transfer To: {config_data['transfer_to']}")
        if config_data.get('escalation_condition'):
            logger.info(f"  - Escalation Condition: {config_data['escalation_condition']}")
        if config_data.get('collection_name'):
            logger.info(f"  - Collection Name: {config_data['collection_name']}")
        
        return config_data
        
    except IOError as e:
        logger.error(f"[ERROR] Failed to write config file: {str(e)}")
        raise IOError(f"Could not write to config file at {CONFIG_FILE}: {str(e)}")
    except Exception as e:
        logger.error(f"[ERROR] Unexpected error updating config: {str(e)}")
        raise


async def update_config_async(
    caller_name: Optional[str] = None,
    agent_instructions: Optional[str] = None,
    tts_language: str = "en",
    voice_id: str = "21m00Tcm4TlvDq8ikWAM",
    additional_params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Async wrapper for update_config to avoid blocking the event loop.
    
    This function runs the synchronous update_config in a thread pool executor
    to ensure it doesn't block async operations.
    
    Args:
        caller_name: Name of the person being called (optional)
        agent_instructions: Custom instructions for the AI agent (optional)
        tts_language: TTS language code (e.g., "en", "es", "fr", "de")
        voice_id: ElevenLabs voice ID (default: "21m00Tcm4TlvDq8ikWAM" - Rachel)
        additional_params: Any additional parameters to store in config (optional)
    
    Returns:
        Dict containing the updated configuration
        
    Raises:
        IOError: If unable to write to config file
        json.JSONDecodeError: If existing config file has invalid JSON
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        update_config,
        caller_name,
        agent_instructions,
        tts_language,
        voice_id,
        additional_params
    )


def load_dynamic_config() -> Dict[str, Any]:
    """
    Load dynamic configuration from config.json file.
    
    This function is called by the agent service at startup or when reconnecting
    to pick up the latest configuration values.
    
    Returns:
        Dict containing the configuration parameters
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        json.JSONDecodeError: If config file has invalid JSON
    """
    try:
        # Check if config file exists
        if not CONFIG_FILE.exists():
            logger.warning(f"Config file not found at {CONFIG_FILE}, creating default config")
            # Create default configuration
            default_config = update_config()
            return default_config
        
        # Read configuration from file
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        
        logger.info(f"[OK] Configuration loaded successfully from {CONFIG_FILE}")
        logger.info(f"  - Caller Name: {config_data.get('caller_name', 'Not set')}")
        logger.info(f"  - TTS Language: {config_data.get('tts_language', 'Not set')}")
        logger.info(f"  - Voice ID: {config_data.get('voice_id', 'Not set')}")
        
        # Log additional parameters if present
        if config_data.get('transfer_to'):
            logger.info(f"  - Transfer To: {config_data.get('transfer_to')}")
        if config_data.get('escalation_condition'):
            logger.info(f"  - Escalation Condition: {config_data.get('escalation_condition')}")
        if config_data.get('collection_name'):
            logger.info(f"  - Collection Name: {config_data.get('collection_name')}")
        
        return config_data
        
    except json.JSONDecodeError as e:
        logger.error(f"[ERROR] Invalid JSON in config file: {str(e)}")
        raise json.JSONDecodeError(
            f"Config file at {CONFIG_FILE} contains invalid JSON",
            e.doc,
            e.pos
        )
    except Exception as e:
        logger.error(f"[ERROR] Error loading config: {str(e)}")
        raise


async def load_dynamic_config_async() -> Dict[str, Any]:
    """
    Async wrapper for load_dynamic_config to avoid blocking the event loop.
    
    This function runs the synchronous load_dynamic_config in a thread pool executor
    to ensure it doesn't block async operations.
    
    Returns:
        Dict containing the configuration parameters
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        json.JSONDecodeError: If config file has invalid JSON
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, load_dynamic_config)


def get_config_value(key: str, default: Any = None) -> Any:
    """
    Get a specific value from the dynamic configuration.
    
    Args:
        key: The configuration key to retrieve
        default: Default value to return if key not found
    
    Returns:
        The configuration value or default if not found
    """
    try:
        config = load_dynamic_config()
        return config.get(key, default)
    except Exception as e:
        logger.warning(f"Could not load config for key '{key}': {str(e)}, using default")
        return default


async def get_config_value_async(key: str, default: Any = None) -> Any:
    """
    Async version of get_config_value.
    
    Args:
        key: The configuration key to retrieve
        default: Default value to return if key not found
    
    Returns:
        The configuration value or default if not found
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, get_config_value, key, default)


# Initialize config file with defaults if it doesn't exist
def initialize_config_if_missing():
    """Initialize config.json with default values if it doesn't exist."""
    if not CONFIG_FILE.exists():
        logger.info("Initializing config.json with default values...")
        update_config()


# Auto-initialize on module import
initialize_config_if_missing()

