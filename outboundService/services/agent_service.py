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
logger.info(f"INSHORA_KNOWLEDGE_BASE: Loaded ({len(INSHORA_KNOWLEDGE_BASE)} characters)")
logger.info("-" * 60)
logger.info("Available Agent Tools:")
logger.info("  - transfer_to_human: Transfer SIP caller to human agent")
logger.info("  - end_call: End call gracefully")
logger.info("  - set_user_action: Set action type (add/update) and insurance type")
logger.info("  - collect_home_insurance_data: Collect home insurance information")
logger.info("  - collect_auto_insurance_data: Collect auto insurance information")
logger.info("  - collect_flood_insurance_data: Collect flood insurance information")
logger.info("  - collect_life_insurance_data: Collect life insurance information")
logger.info("  - collect_commercial_insurance_data: Collect commercial insurance information")
logger.info("  - submit_quote_request: Submit collected insurance quote request")
logger.info("  - create_agencyzoom_lead: Create new lead in AgencyZoom")
logger.info("  - submit_collected_data_to_agencyzoom: Submit all collected data to AgencyZoom")
logger.info(f"Total Tools: 11")
logger.info("=" * 60)


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
        insurance_service: InsuranceService = None,
        agencyzoom_service: AgencyZoomService = None
    ) -> None:
        if instructions is None:
            instructions = os.getenv("AGENT_INSTRUCTIONS", "You are a helpful voice AI assistant.")
        logger.info(f"Agent initialized with instructions: {instructions[:200]}...")
        super().__init__(instructions=instructions)
        
        # Initialize services (create new ones if not provided)
        self.agencyzoom_service = agencyzoom_service or AgencyZoomService()
        self.insurance_service = insurance_service or InsuranceService(agencyzoom_service=self.agencyzoom_service)
        
        # Initialize tool sets
        self.base_tools = BaseTools()
        self.insurance_tools = InsuranceTools(self.insurance_service)
        
        logger.info("Assistant initialized with insurance and AgencyZoom services")

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

    # Expose insurance tools as agent methods for function calling
    @function_tool()
    async def set_user_action(self, action_type: str, insurance_type: str) -> str:
        """Set the user action type (add/update) and insurance type.
        
        Args:
            action_type: Either "add" for new insurance or "update" for existing policy
            insurance_type: Type of insurance - "home", "auto", "flood", "life", or "commercial"
        """
        logger.info(f"🔧 Agent tool called: set_user_action({action_type}, {insurance_type})")
        return self.insurance_service.set_user_action(action_type, insurance_type)
    
    @function_tool()
    async def collect_home_insurance_data(
        self,
        full_name: str,
        date_of_birth: str,
        property_address: str,
        phone: str,
        email: str,
        spouse_name: str = None,
        spouse_dob: str = None,
        has_solar_panels: bool = False,
        has_pool: bool = False,
        roof_age: int = 0,
        has_pets: bool = False,
        current_provider: str = None,
        renewal_date: str = None,
        renewal_premium: float = None
    ) -> str:
        """Collect home insurance information from the caller.
        Args:
            full_name: Full name of primary insured
            date_of_birth: Date of birth (YYYY-MM-DD format)
            property_address: Property address
            phone: Phone number
            email: Email address
            spouse_name: Spouse name (optional)
            spouse_dob: Spouse date of birth (optional, YYYY-MM-DD format)
            has_solar_panels: Whether property has solar panels
            has_pool: Whether property has a pool
            roof_age: Age of roof in years
            has_pets: Whether household has pets
            current_provider: Current insurance provider (optional)
            renewal_date: Current policy renewal date (optional, YYYY-MM-DD format)
            renewal_premium: Current renewal premium amount (optional)
        """
        logger.info(f"🔧 Agent tool called: collect_home_insurance_data({full_name})")
        return self.insurance_service.collect_home_insurance(
            full_name=full_name,
            date_of_birth=date_of_birth,
            spouse_name=spouse_name,
            spouse_dob=spouse_dob,
            property_address=property_address,
            has_solar_panels=has_solar_panels,
            has_pool=has_pool,
            roof_age=roof_age,
            has_pets=has_pets,
            current_provider=current_provider,
            renewal_date=renewal_date,
            renewal_premium=renewal_premium,
            phone=phone,
            email=email
        )
    
    @function_tool()
    async def collect_auto_insurance_data(
        self,
        driver_name: str,
        driver_dob: str,
        license_number: str,
        qualification: str,
        profession: str,
        vin: str,
        vehicle_make: str,
        vehicle_model: str,
        phone: str,
        email: str,
        gpa: float = None,
        coverage_type: str = "full",
        current_provider: str = None,
        renewal_date: str = None,
        renewal_premium: float = None
    ) -> str:
        """Collect auto insurance information from the caller.
        Args:
            driver_name: Full name of driver
            driver_dob: Driver date of birth (YYYY-MM-DD format)
            license_number: Driver's license number
            qualification: Driver qualification
            profession: Driver profession
            vin: Vehicle VIN (17 characters)
            vehicle_make: Vehicle make
            vehicle_model: Vehicle model
            phone: Phone number
            email: Email address
            gpa: GPA if driver under 21 (optional)
            coverage_type: Coverage type - "liability" or "full"
            current_provider: Current insurance provider (optional)
            renewal_date: Current policy renewal date (optional, YYYY-MM-DD format)
            renewal_premium: Current renewal premium amount (optional)
        """
        logger.info(f"🔧 Agent tool called: collect_auto_insurance_data({driver_name})")
        return self.insurance_service.collect_auto_insurance(
            driver_name=driver_name,
            driver_dob=driver_dob,
            license_number=license_number,
            qualification=qualification,
            profession=profession,
            gpa=gpa,
            vin=vin,
            vehicle_make=vehicle_make,
            vehicle_model=vehicle_model,
            coverage_type=coverage_type,
            current_provider=current_provider,
            renewal_date=renewal_date,
            renewal_premium=renewal_premium,
            phone=phone,
            email=email
        )
    
    @function_tool()
    async def collect_flood_insurance_data(self, full_name: str, home_address: str, email: str) -> str:
        """Collect flood insurance information from the caller.
        Args:
            full_name: Full name of insured
            home_address: Home address for flood insurance
            email: Email address
        """
        logger.info(f"🔧 Agent tool called: collect_flood_insurance_data({full_name})")
        return self.insurance_service.collect_flood_insurance(full_name, home_address, email)
    
    @function_tool()
    async def collect_life_insurance_data(
        self,
        full_name: str,
        date_of_birth: str,
        appointment_requested: bool,
        phone: str,
        email: str,
        appointment_date: str = None,
        policy_type: str = None
    ) -> str:
        """Collect life insurance information from the caller.
        Args:
            full_name: Full name of insured
            date_of_birth: Date of birth (YYYY-MM-DD format)
            appointment_requested: Whether customer wants an appointment
            phone: Phone number
            email: Email address
            appointment_date: Requested appointment date and time (optional, YYYY-MM-DD HH:MM format)
            policy_type: Type of policy - "term", "whole", "universal", "annuity", or "long_term_care" (optional)
        """
        logger.info(f"🔧 Agent tool called: collect_life_insurance_data({full_name})")
        return self.insurance_service.collect_life_insurance(
            full_name=full_name,
            date_of_birth=date_of_birth,
            appointment_requested=appointment_requested,
            appointment_date=appointment_date,
            phone=phone,
            email=email,
            policy_type=policy_type
        )
    
    @function_tool()
    async def collect_commercial_insurance_data(
        self,
        business_name: str,
        business_type: str,
        business_address: str,
        phone: str,
        email: str,
        inventory_limit: float = None,
        building_coverage: bool = False,
        building_coverage_limit: float = None,
        current_provider: str = None,
        renewal_date: str = None,
        renewal_premium: float = None
    ) -> str:
        """Collect commercial insurance information from the caller.
        Args:
            business_name: Name of the business
            business_type: Type of business
            business_address: Business address
            phone: Phone number
            email: Email address
            inventory_limit: Inventory coverage limit (optional)
            building_coverage: Whether building coverage is needed
            building_coverage_limit: Building coverage limit (optional)
            current_provider: Current insurance provider (optional)
            renewal_date: Current policy renewal date (optional, YYYY-MM-DD format)
            renewal_premium: Current renewal premium amount (optional)
        """
        logger.info(f"🔧 Agent tool called: collect_commercial_insurance_data({business_name})")
        return self.insurance_service.collect_commercial_insurance(
            business_name=business_name,
            business_type=business_type,
            business_address=business_address,
            inventory_limit=inventory_limit,
            building_coverage=building_coverage,
            building_coverage_limit=building_coverage_limit,
            current_provider=current_provider,
            renewal_date=renewal_date,
            renewal_premium=renewal_premium,
            phone=phone,
            email=email
        )
    
    @function_tool()
    async def submit_quote_request(self) -> str:
        """Submit the collected insurance quote request."""
        logger.info("🔧 Agent tool called: submit_quote_request()")
        return self.insurance_service.submit_quote_request()
    
    
    # AgencyZoom Integration Tools
    @function_tool()
    async def create_agencyzoom_lead(
        self, 
        first_name: str, 
        last_name: str, 
        email: str, 
        phone: str, 
        insurance_type: str,
        notes: str = "",
        address: str = "",
        date_of_birth: str = "",
        current_provider: str = "",
        vehicle_info: str = "",
        property_info: str = "",
        business_name: str = "",
        appointment_requested: bool = False
    ) -> str:
        """Create a new lead in AgencyZoom with detailed information.
        
        Args:
            first_name: Lead's first name
            last_name: Lead's last name
            email: Lead's email address
            phone: Lead's phone number
            insurance_type: Type of insurance interested in (home, auto, flood, life, commercial)
            notes: Additional notes about the lead (optional)
            address: Home or business address (optional)
            date_of_birth: Date of birth for life/personal insurance (optional)
            current_provider: Current insurance provider name (optional)
            vehicle_info: Vehicle details for auto insurance (optional)
            property_info: Property details for home insurance (optional)
            business_name: Business name for commercial insurance (optional)
            appointment_requested: Whether customer wants an appointment (optional)
        
        Returns:
            String message confirming lead creation or error
        """
        logger.info(f"🔧 Agent tool called: create_agencyzoom_lead({first_name} {last_name}, {insurance_type})")
        
        lead_data = {
            "first_name": first_name,
            "last_name": last_name,
            "email": email,
            "phone": phone,
            "insurance_type": insurance_type,
            "notes": notes,
            "source": "AI Phone Call"
        }
        
        # Add optional fields if provided
        if address:
            lead_data["address"] = address
        if date_of_birth:
            lead_data["date_of_birth"] = date_of_birth
        if current_provider:
            lead_data["current_provider"] = current_provider
        if vehicle_info:
            lead_data["vehicle_info"] = vehicle_info
        if property_info:
            lead_data["property_info"] = property_info
        if business_name:
            lead_data["business_name"] = business_name
        if appointment_requested:
            lead_data["appointment_requested"] = appointment_requested
        
        try:
            result = self.agencyzoom_service.create_lead(lead_data)
            if result:
                detail_msg = f"Successfully created lead in AgencyZoom for {first_name} {last_name}. "
                detail_msg += f"They are interested in {insurance_type} insurance."
                if current_provider:
                    detail_msg += f" Current provider: {current_provider}."
                if appointment_requested:
                    detail_msg += " Appointment requested."
                return detail_msg
            else:
                return "Failed to create lead in AgencyZoom. Please check the logs for details."
        except Exception as e:
            logger.error(f"Error creating AgencyZoom lead: {e}")
            return f"Error creating lead: {str(e)}"
    
    
    
    
    
    @function_tool()
    async def submit_collected_data_to_agencyzoom(self) -> str:
        """Submit all collected insurance data to AgencyZoom as a comprehensive lead.
        This automatically extracts all the detailed information collected during the call
        and creates a lead with complete insurance-specific fields.
        
        Returns:
            String message confirming submission or error
        """
        logger.info("🔧 Agent tool called: submit_collected_data_to_agencyzoom()")
        
        # Check if we have insurance data collected
        if not self.insurance_service.insurance_type:
            return "No insurance data has been collected yet. Please collect insurance information first."
        
        insurance_type = self.insurance_service.insurance_type
        insurance_key = f"{insurance_type}_insurance"
        
        if insurance_key not in self.insurance_service.collected_data:
            return f"No {insurance_type} insurance data found. Please collect the information first."
        
        try:
            insurance_data = self.insurance_service.collected_data[insurance_key]
            
            # Extract name and contact info based on insurance type
            first_name = ""
            last_name = ""
            email = ""
            phone = ""
            
            # Extract data based on insurance type
            if insurance_type == "home":
                full_name = insurance_data.get("primary_insured", {}).get("full_name", "")
                if full_name:
                    name_parts = full_name.split(" ", 1)
                    first_name = name_parts[0]
                    last_name = name_parts[1] if len(name_parts) > 1 else ""
                contact_info = insurance_data.get("contact", {})
                email = contact_info.get("email", "")
                phone = contact_info.get("phone", "")
                
            elif insurance_type == "auto":
                drivers = insurance_data.get("drivers", [])
                if drivers:
                    full_name = drivers[0].get("full_name", "")
                    if full_name:
                        name_parts = full_name.split(" ", 1)
                        first_name = name_parts[0]
                        last_name = name_parts[1] if len(name_parts) > 1 else ""
                contact_info = insurance_data.get("contact", {})
                email = contact_info.get("email", "")
                phone = contact_info.get("phone", "")
                
            elif insurance_type == "flood":
                full_name = insurance_data.get("full_name", "")
                if full_name:
                    name_parts = full_name.split(" ", 1)
                    first_name = name_parts[0]
                    last_name = name_parts[1] if len(name_parts) > 1 else ""
                email = insurance_data.get("email", "")
                phone = insurance_data.get("phone", "")
                
            elif insurance_type == "life":
                full_name = insurance_data.get("insured", {}).get("full_name", "")
                if full_name:
                    name_parts = full_name.split(" ", 1)
                    first_name = name_parts[0]
                    last_name = name_parts[1] if len(name_parts) > 1 else ""
                contact_info = insurance_data.get("contact", {})
                email = contact_info.get("email", "")
                phone = contact_info.get("phone", "")
                
            elif insurance_type == "commercial":
                business_name = insurance_data.get("business", {}).get("name", "")
                first_name = business_name  # For commercial, use business name
                last_name = ""
                contact_info = insurance_data.get("contact", {})
                email = contact_info.get("email", "")
                phone = contact_info.get("phone", "")
            
            # Create comprehensive lead data
            lead_data = {
                "first_name": first_name or "Unknown",
                "last_name": last_name or "",
                "email": email or "noemail@pending.com",
                "phone": phone or "",
                "insurance_type": insurance_type,
                "source": "AI Voice Agent",
                "notes": f"Lead collected via AI voice agent. Session ID: {self.insurance_service.session_id}",
                "insurance_details": insurance_data  # Include ALL detailed insurance data
            }
            
            # Add insurance-type specific fields to top level for easy access
            if insurance_type == "home":
                lead_data["property_address"] = insurance_data.get("property", {}).get("address", "")
                lead_data["has_pool"] = insurance_data.get("property", {}).get("has_pool", False)
                lead_data["has_solar_panels"] = insurance_data.get("property", {}).get("has_solar_panels", False)
                lead_data["roof_age"] = insurance_data.get("property", {}).get("roof_age", 0)
                lead_data["has_pets"] = insurance_data.get("has_pets", False)
                lead_data["current_provider"] = insurance_data.get("current_policy", {}).get("current_provider", "")
                
            elif insurance_type == "auto":
                vehicles = insurance_data.get("vehicles", [])
                if vehicles:
                    lead_data["vehicle_vin"] = vehicles[0].get("vin", "")
                    lead_data["vehicle_make"] = vehicles[0].get("make", "")
                    lead_data["vehicle_model"] = vehicles[0].get("model", "")
                    lead_data["coverage_type"] = vehicles[0].get("coverage_type", "")
                drivers = insurance_data.get("drivers", [])
                if drivers:
                    lead_data["license_number"] = drivers[0].get("license_number", "")
                    lead_data["profession"] = drivers[0].get("profession", "")
                lead_data["current_provider"] = insurance_data.get("current_policy", {}).get("current_provider", "")
                
            elif insurance_type == "flood":
                lead_data["home_address"] = insurance_data.get("home_address", "")
                
            elif insurance_type == "life":
                lead_data["appointment_requested"] = insurance_data.get("appointment_requested", False)
                lead_data["appointment_date"] = insurance_data.get("appointment_date", "")
                lead_data["policy_type"] = insurance_data.get("policy_type", "")
                lead_data["date_of_birth"] = insurance_data.get("insured", {}).get("date_of_birth", "")
                
            elif insurance_type == "commercial":
                lead_data["business_name"] = insurance_data.get("business", {}).get("name", "")
                lead_data["business_type"] = insurance_data.get("business", {}).get("type", "")
                lead_data["business_address"] = insurance_data.get("business", {}).get("address", "")
                lead_data["inventory_limit"] = insurance_data.get("coverage", {}).get("inventory_limit", "")
                lead_data["building_coverage"] = insurance_data.get("coverage", {}).get("building_coverage", False)
                lead_data["current_provider"] = insurance_data.get("current_policy", {}).get("current_provider", "")
            
            # Submit to AgencyZoom
            result = self.agencyzoom_service.create_lead(lead_data)
            
            if result:
                logger.info(f"Successfully submitted comprehensive {insurance_type} insurance data to AgencyZoom")
                return f"Excellent! I've successfully submitted all your {insurance_type} insurance information to AgencyZoom with complete details including all the specific information you provided. Our team will follow up with you shortly!"
            else:
                return "Failed to submit data to AgencyZoom. The information is saved locally and can be submitted manually."
                
        except Exception as e:
            logger.error(f"Error submitting collected data to AgencyZoom: {e}", exc_info=True)
            return f"Error submitting to AgencyZoom: {str(e)}. The data is still saved locally."

    
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
        
        dynamic_config = load_dynamic_config()
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
        
        # Add the Inshora Knowledge Base and tool instructions
        instructions = f"""{base_instructions}

{INSHORA_KNOWLEDGE_BASE}

USE THIS KNOWLEDGE BASE TO:
- Answer questions about Texas insurance requirements accurately
- Handle objections professionally using the provided scripts
- Adapt your tone based on the caller's communication style
- Mention relevant promotions and discounts when appropriate
- Cross-sell based on the lead scoring matrix
- Know when to escalate to a human agent
- Use rebuttals when needed to keep the conversation productive

AVAILABLE TOOLS - Use these during the conversation:

1. CALL MANAGEMENT:
   - transfer_to_human: Transfer the caller to a human agent when escalation is needed
   - end_call: End the call gracefully when conversation is complete

2. INSURANCE DATA COLLECTION:
   - set_user_action: FIRST call this to set action type ("add" or "update") and insurance type ("home", "auto", "flood", "life", "commercial")
   - collect_home_insurance_data: Collect home insurance details (name, DOB, address, phone, email, solar panels, pool, roof age, pets, current provider)
   - collect_auto_insurance_data: Collect auto insurance details (driver info, license, VIN, vehicle make/model, coverage type)
   - collect_flood_insurance_data: Collect flood insurance details (name, address, email)
   - collect_life_insurance_data: Collect life insurance details (name, DOB, appointment request, policy type)
   - collect_commercial_insurance_data: Collect commercial insurance details (business name, type, address, inventory limit, building coverage)
   - submit_quote_request: Submit the collected insurance data for quote processing

3. CRM INTEGRATION:
   - create_agencyzoom_lead: Create a new lead in AgencyZoom with customer details
   - submit_collected_data_to_agencyzoom: Submit ALL collected insurance data to AgencyZoom as a comprehensive lead

WORKFLOW:
1. Greet the caller and identify their insurance needs
2. Call set_user_action with the appropriate action and insurance type
3. Use the relevant collect_*_insurance_data tool to gather information
4. Call submit_quote_request to process the quote
5. Call submit_collected_data_to_agencyzoom to save lead to CRM
6. If caller requests human assistance or meets escalation condition, use transfer_to_human"""
        
        # Add escalation condition if provided
        if escalation_condition:
            instructions = f"{instructions}\n\nESCALATION CONDITION: {escalation_condition}"
        
        
        logger.info("✓ Dynamic configuration loaded successfully")
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
        instructions = f"""You are a professional AI insurance assistant for Inshora Group.

{INSHORA_KNOWLEDGE_BASE}

USE THIS KNOWLEDGE BASE TO:
- Answer questions about Texas insurance requirements accurately
- Handle objections professionally using the provided scripts
- Adapt your tone based on the caller's communication style
- Mention relevant promotions and discounts when appropriate
- Cross-sell based on the lead scoring matrix
- Know when to escalate to a human agent
- Use rebuttals when needed to keep the conversation productive

AVAILABLE TOOLS - Use these during the conversation:

1. CALL MANAGEMENT:
   - transfer_to_human: Transfer the caller to a human agent when escalation is needed
   - end_call: End the call gracefully when conversation is complete

2. INSURANCE DATA COLLECTION:
   - set_user_action: FIRST call this to set action type ("add" or "update") and insurance type ("home", "auto", "flood", "life", "commercial")
   - collect_home_insurance_data: Collect home insurance details (name, DOB, address, phone, email, solar panels, pool, roof age, pets, current provider)
   - collect_auto_insurance_data: Collect auto insurance details (driver info, license, VIN, vehicle make/model, coverage type)
   - collect_flood_insurance_data: Collect flood insurance details (name, address, email)
   - collect_life_insurance_data: Collect life insurance details (name, DOB, appointment request, policy type)
   - collect_commercial_insurance_data: Collect commercial insurance details (business name, type, address, inventory limit, building coverage)
   - submit_quote_request: Submit the collected insurance data for quote processing

3. CRM INTEGRATION:
   - create_agencyzoom_lead: Create a new lead in AgencyZoom with customer details
   - submit_collected_data_to_agencyzoom: Submit ALL collected insurance data to AgencyZoom as a comprehensive lead

WORKFLOW:
1. Greet the caller and identify their insurance needs
2. Call set_user_action with the appropriate action and insurance type
3. Use the relevant collect_*_insurance_data tool to gather information
4. Call submit_quote_request to process the quote
5. Call submit_collected_data_to_agencyzoom to save lead to CRM
6. If caller requests human assistance, use transfer_to_human"""
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
    
    async def cleanup_and_save():
        """
        Non-blocking cleanup that runs in background.
        This ensures participant disconnect doesn't block the server.
        """
        try:
            logger.info("Cleanup started (non-blocking)...")
            
            # session may not be defined if start() failed — guard it
            if "session" in locals() and session is not None and hasattr(session, "history"):
                transcript_data = session.history.to_dict()
                
                # Save to MongoDB in background (don't block disconnect)
                try:
                    # Get caller information from dynamic config (use cached version)
                    logger.info("Saving transcript to MongoDB...")
                    dynamic_config = load_dynamic_config()
                    caller_name = dynamic_config.get("caller_name", "Guest")
                    contact_number = dynamic_config.get("contact_number")
                    
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
                        logger.info(f"Transcript saved to MongoDB with ID: {transcript_id}")
                    else:
                        logger.warning("MONGODB_URI not set, skipping MongoDB transcript save")
                except Exception as mongo_error:
                    logger.error(f"Failed to save transcript to MongoDB: {mongo_error}", exc_info=True)
                    # Don't fail the cleanup if MongoDB save fails
            else:
                logger.warning("No session history to save (session not created or no history).")
            
            logger.info("Cleanup completed successfully")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}", exc_info=True)

    # Wrap cleanup in background task to avoid blocking server on disconnect
    async def cleanup_wrapper():
        """Non-blocking wrapper that schedules cleanup as background task"""
        # Schedule cleanup in background without waiting for it
        asyncio.create_task(cleanup_and_save())
        logger.info("[OK] Cleanup task scheduled (non-blocking)")
        # Return immediately - don't wait for cleanup to finish
    
    ctx.add_shutdown_callback(cleanup_wrapper)
    logger.info("[OK] Shutdown callback added (non-blocking)")

    # --------------------------------------------------------
    # Initialize core components
    # --------------------------------------------------------
    try:
        logger.info("Initializing session components...")

        logger.info("Step 1: Initializing STT (Deepgram)")
        stt_instance = deepgram.STT(model=STT_MODEL, language=STT_LANGUAGE)

        logger.info(f"Step 2: Initializing LLM ({provider})")
        
        # Initialize LLM based on provider from config
        if provider == "gemini":
            if api_key:
                logger.info("Using Gemini with custom API key")
                # Set the API key in environment for Google plugin
                os.environ["GOOGLE_API_KEY"] = api_key
                llm_instance = google.LLM(model="gemini-2.5-pro")
                logger.info("[OK] Gemini LLM initialized with custom API key")
            else:
                logger.warning("Gemini provider selected but no API key provided, falling back to OpenAI")
                llm_instance = openai.LLM(model=LLM_MODEL)
                logger.info("[OK] OpenAI LLM initialized (fallback)")
        else:  # default to OpenAI
            if api_key:
                logger.info("Using OpenAI with custom API key")
                # Set the API key in environment for OpenAI plugin
                os.environ["OPENAI_API_KEY"] = api_key
                llm_instance = openai.LLM(model="gpt-4.1-mini")
                logger.info("[OK] OpenAI LLM initialized with custom API key")
            else:
                logger.info("Using default OpenAI configuration")
                llm_instance = openai.LLM(model=LLM_MODEL)
                logger.info("[OK] OpenAI LLM initialized with default config")

        logger.info("Step 3: Initializing TTS (ElevenLabs)")
        try:
            tts_instance = elevenlabs.TTS(
                base_url="https://api.eu.residency.elevenlabs.io/v1",
                voice_id=voice_id,
                language=language,
                model="eleven_flash_v2_5"
            )
        #     tts_instance = cartesia.TTS(
        #     model='sonic-3',
        #     voice='a0e99841-438c-4a64-b679-ae501e7d6091',
        #     language='en',
        #     speed=1.0,
        #     sample_rate=24000
        # )

            logger.info("[OK] ElevenLabs TTS initialized successfully")
        except Exception as tts_error:
            logger.warning(f"ElevenLabs TTS initialization failed: {tts_error}")
            logger.info("Falling back to OpenAI TTS...")
            # Fallback to OpenAI TTS
            tts_instance = openai.TTS(
                voice="alloy",
                model="tts-1"
            )
            logger.info("[OK] OpenAI TTS initialized as fallback")

        logger.info("Step 4: Creating AgentSession")
        session = AgentSession(vad=silero.VAD.load(),stt=stt_instance, llm=llm_instance, tts=tts_instance)
        logger.info("[OK] All session components initialized")
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
