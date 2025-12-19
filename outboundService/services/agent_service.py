import os
import json
import asyncio
import logging
import sys
import aiohttp
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from pathlib import Path
from typing import Optional
from livekit import api
from livekit import agents
from livekit.agents import AgentSession, Agent, RoomInputOptions, function_tool, RunContext, get_job_context
from livekit.plugins import (
    openai,
    cartesia,
    deepgram,
    silero,
    google
)
from livekit.plugins.openai.realtime.realtime_model import TurnDetection
# from voice_backend.outboundService.common.config.settings import ROOM_NAME
from dotenv import load_dotenv
from common.config.settings import (
    TTS_MODEL, TTS_VOICE, STT_MODEL, STT_LANGUAGE, LLM_MODEL, TRANSCRIPT_DIR, PARTICIPANT_IDENTITY
)
from common.update_config import load_dynamic_config
from livekit.plugins import elevenlabs

# Add project root to path for database imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
from database.mongo import get_mongodb_manager

# Import knowledge base for voice agent
from config.knowledgebase import get_knowledge_base

# Import services for insurance data collection
from services.insurance_service import InsuranceService
from services.agencyzoom import AgencyZoomService
from tools.base_tools import BaseTools
from tools.insurance_tools import InsuranceTools

# Load knowledge base once at module level
INSHORA_KNOWLEDGE_BASE = get_knowledge_base()

load_dotenv()

# ------------------------------------------------------------
# Environment / LiveKit admin credentials (fetched from env)
# ------------------------------------------------------------
LIVEKIT_URL = os.getenv("LIVEKIT_URL")
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET")



# Tools registry path
TOOLS_FILE = Path(__file__).parent.parent.parent.parent / "tools.json"

# Global caches to avoid repeated file I/O during conversation
_TOOLS_CACHE = None
_DYNAMIC_CONFIG_CACHE = None
_CACHE_TIMESTAMP = 0
CACHE_TTL = 60  # Cache for 60 seconds

# ------------------------------------------------------------
# Logging setup
# ------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("agent_debug.log")
    ]
)
logger = logging.getLogger("services.agent_service")

logger.info("=" * 60)
logger.info("Agent Service Module Loading")
logger.info(f"LIVEKIT_URL: {LIVEKIT_URL or 'NOT SET'}")
logger.info(f"LIVEKIT_API_KEY: {LIVEKIT_API_KEY}")
logger.info(f"LIVEKIT_API_SECRET: {LIVEKIT_API_SECRET}")
logger.info(f"OPENAI_API_KEY: {'SET' if os.getenv('OPENAI_API_KEY') else 'NOT SET'}")
logger.info(f"STT_MODEL: {STT_MODEL}")
logger.info(f"LLM_MODEL: {LLM_MODEL}")


# ------------------------------------------------------------
# Utility: cleanup previous rooms with safe guards
# ------------------------------------------------------------
async def cleanup_previous_rooms(api_key, api_secret, server_url, prefix="agent-room"):
    """
    Attempt to delete previously created rooms whose name starts with `prefix`.
    This function is defensive: if the admin API isn't available it logs and continues.
    """
    if not (api_key and api_secret and server_url):
        logger.warning("LiveKit admin credentials or URL not provided — skipping room cleanup.")
        return

    try:
        logger.info("Attempting to list & cleanup previous rooms (prefix=%s)...", prefix)
        # Use LiveKitAPI for room management
        lk_api = api.LiveKitAPI(url=server_url, api_key=api_key, api_secret=api_secret)
        room_service = lk_api.room
        
        # List all rooms
        active_rooms = await room_service.list_rooms(api.ListRoomsRequest())
        
        # active_rooms may be an object with .rooms or be a list depending on SDK
        rooms_iterable = getattr(active_rooms, "rooms", active_rooms)
        deleted = 0
        for room in rooms_iterable:
            name = getattr(room, "name", None) or room
            if name and name.startswith(prefix):
                logger.info("Room cleanup: Deleting old room: %s", name)
                try:
                    # Delete the room using the request object
                    await room_service.delete_room(api.DeleteRoomRequest(room=name))
                    deleted += 1
                except Exception as e:
                    logger.warning("Failed to delete room %s: %s", name, e)
        
        # Close the API connection
        await lk_api.aclose()
        logger.info("Room cleanup finished - deleted %d rooms matching prefix '%s'", deleted, prefix)
    except Exception as e:
        logger.warning("Room cleanup failed (non-fatal). Reason: %s", e, exc_info=True)


# ------------------------------------------------------------
# Assistant definition
# ------------------------------------------------------------
class Assistant(Agent):
    def __init__(
        self, 
        instructions: str = None,
       
    ) -> None:
        if instructions is None:
            instructions = os.getenv("AGENT_INSTRUCTIONS", "You are a helpful voice AI assistant.")
        logger.info(f"Agent initialized with instructions: {instructions[:200]}...")
        super().__init__(instructions=instructions)
        
        

    @function_tool
    async def transfer_to_human(self, ctx: RunContext) -> str:
        """Transfer active SIP caller to a human number. if it satisfies the escalation condition, transfer to the human number."""
        job_ctx = get_job_context()
        if job_ctx is None:
            logger.error("Job context not found")
            return "error"
        
        # Load transfer_to from dynamic config (using cache to avoid blocking I/O)
        global _DYNAMIC_CONFIG_CACHE, _CACHE_TIMESTAMP
        import time
        
        current_time = time.time()
        
        # Refresh cache if expired
        if _DYNAMIC_CONFIG_CACHE is None or (current_time - _CACHE_TIMESTAMP) >= CACHE_TTL:
            _DYNAMIC_CONFIG_CACHE = load_dynamic_config()
            _CACHE_TIMESTAMP = current_time
        
        transfer_to_number = _DYNAMIC_CONFIG_CACHE.get("transfer_to", "+919911062767")
        
        # Ensure the transfer_to number has the "tel:" prefix for SIP
        if not transfer_to_number.startswith("tel:"):
            transfer_to = f"tel:{transfer_to_number}"
        else:
            transfer_to = transfer_to_number
        
        logger.info(f"Transfer requested to: {transfer_to}")

        sip_participant = None
        for participant in job_ctx.room.remote_participants.values():
            if participant.identity == "sip-caller":
                sip_participant = participant
                break

        if sip_participant is None:
            logger.error("No SIP participant found to transfer")
            return "error"

        try:
            await job_ctx.api.sip.transfer_sip_participant(
                api.TransferSIPParticipantRequest(
                    room_name=job_ctx.room.name,
                    participant_identity=sip_participant.identity,
                    transfer_to=transfer_to,
                    play_dialtone=True
                )
            )
            logger.info(f"Transferred participant {sip_participant.identity} to {transfer_to}")
            return "transferred"
        except Exception as e:
            logger.error(f"Failed to transfer call: {e}", exc_info=True)
            return "error"

    @function_tool
    async def end_call(self, ctx: RunContext) -> str:
        """End call gracefully."""
        logger_local = logging.getLogger("phone-assistant")
        job_ctx = get_job_context()
        if job_ctx is None:
            logger_local.error("Failed to get job context")
            return "error"

        try:
            await job_ctx.api.room.delete_room(api.DeleteRoomRequest(room=job_ctx.room.name))
            logger_local.info(f"Successfully ended call for room {job_ctx.room.name}")
            return "ended"
        except Exception as e:
            logger_local.error(f"Failed to end call: {e}", exc_info=True)
            return "error"

    
# ------------------------------------------------------------
# Agent entrypoint
# ------------------------------------------------------------
async def entrypoint(ctx: agents.JobContext):
    """Main entrypoint for the voice agent service."""
    logger.info("=" * 60)
    logger.info(f"ENTRYPOINT CALLED - Room: {ctx.room.name}")
    logger.info("=" * 60)

    # Load dynamic configuration from config.json
    try:
        logger.info("Loading dynamic configuration from config.json...")
        
        # Load and store in global cache
        global _DYNAMIC_CONFIG_CACHE, _CACHE_TIMESTAMP
        import time
        
        dynamic_config = load_dynamic_config()
        _DYNAMIC_CONFIG_CACHE = dynamic_config
        _CACHE_TIMESTAMP = time.time()
        logger.info("✓ Dynamic config loaded and cached globally")
        
        caller_name = dynamic_config.get("caller_name", "Guest")
        dynamic_instruction = dynamic_config.get("agent_instructions", "You are a helpful voice AI assistant.")
        language = dynamic_config.get("tts_language", "en")
        voice_id = dynamic_config.get("voice_id", "21m00Tcm4TlvDq8ikWAM")
        escalation_condition = dynamic_config.get("escalation_condition")
        provider = dynamic_config.get("provider", "openai").lower()
        api_key = dynamic_config.get("api_key")

        # Build full instructions with knowledge base and escalation condition
        # Start with the dynamic instruction
        base_instructions = dynamic_instruction
        
        
        # Add escalation condition if provided
        if escalation_condition:
            instructions = f"{base_instructions}\n\nESCALATION CONDITION strictly follow this condition and transfer the call to the human agent: {escalation_condition}"
        
        
        logger.info("Dynamic configuration loaded successfully")
        logger.info(f"  - Caller Name: {caller_name}")
        logger.info(f"  - TTS Language: {language}")
        logger.info(f"  - Voice ID: {voice_id}")
        logger.info(f"  - LLM Provider: {provider}")
        if api_key:
            logger.info(f"  - Custom API Key: {'***' + api_key[-4:] if len(api_key) > 4 else '***'}")
        logger.info(f"  - Agent Instructions: {dynamic_instruction[:100]}...")
        if escalation_condition:
            logger.info(f"  - Escalation Condition: {escalation_condition}")
        
        # Log full instructions to see what AI knows (first 500 chars)
        logger.info(f"  - Full Instructions Preview: {instructions[:500]}...")
        if len(instructions) > 500:
            logger.info(f"  - Total Instruction Length: {len(instructions)} characters")
        
   
    except Exception as e:
        logger.warning(f"Failed to load dynamic config, using defaults: {str(e)}")
        caller_name = "Guest"
        # Include knowledge base in fallback instructions
        instructions = f"""You are a professional AI insurance assistant for Inshora Group."""
        language = "en"
        voice_id = "21m00Tcm4TlvDq8ikWAM"
        provider = "openai"
        api_key = None
    
    # Static config from environment
    room_prefix_for_cleanup = os.getenv("ROOM_CLEANUP_PREFIX", "agent-room")

    # --------------------------------------------------------
    # Prepare cleanup callback (save transcript and clean resources)
    # --------------------------------------------------------
    # Track session start time for duration calculation
    session_start_time = None
    # Variables to hold component references for cleanup
    session = None
    tts_instance = None
    stt_instance = None
    
    async def cleanup_and_save():
        """
        Background task to save transcript and perform non-critical cleanup.
        Critical resource cleanup (session, audio) is done synchronously in cleanup_wrapper.
        """
        nonlocal session, session_start_time
        
        try:
            logger.info("Background cleanup started (transcript saving)...")
            
            # Save transcript if available
            if session is not None and hasattr(session, "history"):
                try:
                    transcript_data = session.history.to_dict()
                    
                    # Get caller information from dynamic config (use cached version)
                    logger.info("Saving transcript to MongoDB...")
                    
                    # Use cached config instead of reloading
                    global _DYNAMIC_CONFIG_CACHE
                    if _DYNAMIC_CONFIG_CACHE is None:
                        _DYNAMIC_CONFIG_CACHE = load_dynamic_config()
                    
                    caller_name = _DYNAMIC_CONFIG_CACHE.get("caller_name", "Guest")
                    contact_number = _DYNAMIC_CONFIG_CACHE.get("contact_number")
                    
                    # Generate caller_id from room name
                    caller_id = ctx.room.name if ctx.room else f"call_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    
                    # Calculate call duration
                    duration_seconds = None
                    if session_start_time is not None:
                        end_time = datetime.utcnow()
                        duration_delta = end_time - session_start_time
                        duration_seconds = int(duration_delta.total_seconds())
                        logger.info(f"Call duration: {duration_seconds} seconds ({duration_delta})")
                    
                    # Get MongoDB manager (singleton - don't close it)
                    mongodb_uri = os.getenv("MONGODB_URI")
                    if mongodb_uri:
                        try:
                            # Add timeout to MongoDB operations to prevent blocking
                            async def save_to_mongo():
                                mongo_manager = get_mongodb_manager(mongodb_uri)
                                
                                # Build metadata with duration
                                metadata = {
                                    "room_name": ctx.room.name if ctx.room else None,
                                    "timestamp": datetime.utcnow().isoformat()
                                }
                                if duration_seconds is not None:
                                    metadata["duration_seconds"] = duration_seconds
                                    metadata["duration_formatted"] = f"{duration_seconds // 60}m {duration_seconds % 60}s"
                                
                                transcript_id = mongo_manager.save_transcript(
                                    transcript=transcript_data,
                                    caller_id=caller_id,
                                    name=caller_name,
                                    contact_number=contact_number,
                                    metadata=metadata
                                )
                                return transcript_id
                            
                            # Set a timeout of 5 seconds for MongoDB save
                            transcript_id = await asyncio.wait_for(save_to_mongo(), timeout=5.0)
                            logger.info(f"Transcript saved to MongoDB with ID: {transcript_id}")
                            
                        except asyncio.TimeoutError:
                            logger.error("MongoDB save operation timed out after 5 seconds - continuing cleanup")
                        except Exception as mongo_error:
                            logger.error(f"MongoDB connection/save error: {mongo_error} - continuing cleanup")
                    else:
                        logger.warning("MONGODB_URI not set, skipping MongoDB transcript save")
                except Exception as mongo_error:
                    logger.error(f"Failed to save transcript to MongoDB: {mongo_error}", exc_info=True)
                    # Don't fail the cleanup if MongoDB save fails
            else:
                logger.warning("No session history to save (session not created or no history).")
            
            logger.info("Background cleanup completed successfully")
        except Exception as e:
            logger.error(f"Error during background cleanup: {e}", exc_info=True)

    # Wrap cleanup in background task to avoid blocking server on disconnect
    async def cleanup_wrapper():
        """
        Wrapper that performs critical cleanup synchronously and schedules rest in background.
        This ensures audio resources are released before returning.
        """
        try:
            # Step 1: Critical cleanup (audio resources) - do this synchronously
            logger.info("Starting critical resource cleanup (synchronous)...")
            
            # Close session to release audio streams
            if session is not None:
                try:
                    if hasattr(session, 'close'):
                        await session.close()
                    elif hasattr(session, 'aclose'):
                        await session.aclose()
                    logger.info("Session closed (critical)")
                except Exception as e:
                    logger.warning(f"Session close in wrapper: {e}")
            
            # Close audio components
            if tts_instance is not None:
                try:
                    if hasattr(tts_instance, 'aclose'):
                        await tts_instance.aclose()
                    elif hasattr(tts_instance, 'close'):
                        tts_instance.close()
                    logger.info("TTS closed (critical)")
                except Exception as e:
                    logger.warning(f"TTS close in wrapper: {e}")
            
            if stt_instance is not None:
                try:
                    if hasattr(stt_instance, 'aclose'):
                        await stt_instance.aclose()
                    elif hasattr(stt_instance, 'close'):
                        stt_instance.close()
                    logger.info("STT closed (critical)")
                except Exception as e:
                    logger.warning(f"STT close in wrapper: {e}")
            
            logger.info("[OK] Critical resources released")
            
            # Step 2: Schedule non-critical cleanup (transcript saving) in background
            asyncio.create_task(cleanup_and_save())
            logger.info("[OK] Non-critical cleanup task scheduled (background)")
        except Exception as e:
            logger.error(f"Error in cleanup wrapper: {e}", exc_info=True)
    
    ctx.add_shutdown_callback(cleanup_wrapper)
    logger.info("[OK] Shutdown callback added (hybrid sync/async)")

    # --------------------------------------------------------
    # Initialize core components
    # --------------------------------------------------------
    try:
        logger.info("Initializing session components with OpenAI Realtime...")
        
        # Add small delay to ensure previous call cleanup is complete
        logger.info("Waiting for any previous cleanup to complete...")
        await asyncio.sleep(0.5)
        logger.info("Step 1: Creating AgentSession with OpenAI Realtime model")
        session = AgentSession(
            # Voice Activity Detection
            # vad=silero.VAD.load(),
            # Create the realtime model
            llm = openai.realtime.RealtimeModel(
                api_key=os.getenv("OPENAI_API_KEY"),
                voice=voice_id,
                model="gpt-realtime",
                temperature=0.2,
                turn_detection=TurnDetection(
                    type="server_vad",
                    threshold=0.6,  # Increased from 0.5 - less sensitive = fewer false triggers
                    prefix_padding_ms=200,  # Reduced from 300 - faster detection
                    silence_duration_ms=400,  # Slightly increased - prevents cutting off user
                ),
                max_session_duration=1800,
            )
            # # Speech-to-Text - Deepgram Nova-3
            # stt=stt_instance, 
            # # Large Language Model
            # llm=llm_instance, 
            # # Text-to-Speech - ElevenLabs
            # tts=tts_instance
        )
        logger.info("[OK] AgentSession with OpenAI Realtime initialized")
    except Exception as e:
        logger.error(f"[ERROR] Failed initializing session components: {e}", exc_info=True)
        raise

    # --------------------------------------------------------
    # Optional: cleanup previous rooms BEFORE connecting
    # --------------------------------------------------------
    try:
        # await cleanup_previous_rooms(LIVEKIT_API_KEY, LIVEKIT_API_SECRET, LIVEKIT_URL, prefix=room_prefix_for_cleanup)
        pass
    except Exception as e:
        logger.warning("cleanup_previous_rooms raised an exception (non-fatal): %s", e, exc_info=True)

    # --------------------------------------------------------
    # Connect to room
    # --------------------------------------------------------
    try:
        logger.info("Connecting to room...")
        await ctx.connect()
        logger.info("[OK] Connected to room successfully")
    except Exception as e:
        logger.error("Failed to connect to room: %s", e, exc_info=True)
        # If connection fails, raise so the worker can restart or exit cleanly
        raise

    # --------------------------------------------------------
    # Initialize assistant and start session
    # --------------------------------------------------------
    assistant = Assistant(instructions=instructions)
    room_options = RoomInputOptions()

    try:
        logger.info("Starting agent session...")
        
        # Track session start time for duration calculation
        session_start_time = datetime.utcnow()
        logger.info(f"Session start time recorded: {session_start_time.isoformat()}")

        await session.start(room=ctx.room, agent=assistant, room_input_options=room_options)
        logger.info("[OK] Agent session started successfully")
    except Exception as e:
        logger.error("Failed to start AgentSession: %s", e, exc_info=True)
        # ensure we attempt a graceful shutdown/cleanup
        try:
            await ctx.shutdown()
        except Exception:
            pass
        raise

    # --------------------------------------------------------
    # Greeting logic AFTER session start and stream stabilization
    # --------------------------------------------------------
    # await asyncio.sleep(2)  # Let audio streams stabilize

    # Multi-language greeting support
    greetings = {
        "en": f"Hello {caller_name}, I'm  Insurance Assistant from Inshora Group.",
    }
    
    # Default to English if language not supported
    greeting_instruction = greetings.get(language, greetings["en"])
    try:
        # Guard that session is running (some SDKs expose is_running)
        is_running = getattr(session, "is_running", None)
        if is_running is None or is_running:
            await session.generate_reply(instructions=greeting_instruction)
            logger.info("[OK] Greeting sent successfully")
        else:
            logger.warning("Session reports not running — skipping greeting.")
    except Exception as e:
        logger.error(f"[ERROR] Failed sending greeting: {e}", exc_info=True)

    # --------------------------------------------------------
    # Wait for shutdown
    # --------------------------------------------------------
    logger.info("Session running — waiting for termination signal...")
    try:
        # Wait for the job context to terminate (standard in livekit-agents 1.2+)
        # await ctx.wait_for_termination()
        logger.info("Termination signal received")
    except asyncio.CancelledError:
        logger.info("Session cancelled - shutting down gracefully")
    except Exception as e:
        logger.error(f"[ERROR] Error while waiting for shutdown: {e}", exc_info=True)
    finally:
        # Fast cleanup - don't block the server
        logger.info("Initiating resource cleanup...")
        
        logger.info("=" * 60)
        logger.info(f"ENTRYPOINT FINISHED - Room: {ctx.room.name}")
        logger.info("=" * 60)


# ------------------------------------------------------------
# CLI entrypoint
# ------------------------------------------------------------
def run_agent():
    """Run the agent CLI worker."""
    logger.info("=" * 60)
    logger.info("RUN_AGENT CALLED - Starting LiveKit Agent CLI")
    logger.info("=" * 60)
    
    # Get agent name from environment or use default
    # agent_name = "voice-assistant"
    # logger.info(f"Starting agent with name: {agent_name}")
    logger.info(f"Agent will listen for new rooms and auto-dispatch")
    logger.info(f"Agent will run CONTINUOUSLY - press Ctrl+C to stop")
    logger.info("=" * 60)
    try:
        # Configure worker to auto-join ALL new rooms
        # When only entrypoint_fnc is provided, it auto-accepts all job requests
        worker_options = agents.WorkerOptions(
            entrypoint_fnc=entrypoint,
        )
        
        logger.info("Worker configured to auto-join ALL new rooms")
        agents.cli.run_app(worker_options)
        logger.info("Agent CLI exited normally")
    except KeyboardInterrupt:
        logger.info("Agent stopped by user (Ctrl+C)")
    except Exception as e:
        logger.error(f"[ERROR] Fatal error in run_agent: {e}", exc_info=True)
        raise
