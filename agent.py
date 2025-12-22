"""Main telephony agent for insurance quote collection."""

import logging
import os
from datetime import datetime
from dotenv import load_dotenv
from livekit.plugins.openai.realtime.realtime_model import TurnDetection


from livekit.agents import (
    cli,
    WorkerOptions,
    JobContext,
    AgentSession,
    RunContext,
)
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    WorkerOptions,
    cli,
    function_tool,
)
from livekit.plugins import deepgram, openai, cartesia, silero
from livekit.plugins import elevenlabs


# from RAGService import RAGService
from services.insurance_service import InsuranceService
from services.ams360 import AMS360Service
from services.agencyzoom import AgencyZoomService
from tools.base_tools import BaseTools
from tools.insurance_tools import InsuranceTools
from config import (
    AgentConfig,
    default_config,
    AGENT_SYSTEM_INSTRUCTIONS,
    get_greeting_prompt
)
from config.knowledgebase import get_knowledge_base

# Load knowledge base once at module level
INSHORA_KNOWLEDGE_BASE = get_knowledge_base()

load_dotenv()
logger = logging.getLogger("telephony-agent")

# Load configuration
config = AgentConfig(
    rag=default_config.rag
)
config.rag.qdrant_url = os.getenv("QDRANT_URL", config.rag.qdrant_url)
config.rag.qdrant_api_key = os.getenv("QDRANT_API_KEY", config.rag.qdrant_api_key)
config.rag.openai_api_key = os.getenv("OPENAI_API_KEY", config.rag.openai_api_key)

# Initialize RAG Service
# rag_service = RAGService(
#     qdrant_url=config.rag.qdrant_url,
#     qdrant_api_key=config.rag.qdrant_api_key,
#     openai_api_key=config.rag.openai_api_key
# )


class TelephonyAgent(Agent):
    """Enhanced Telephony Agent with Insurance Quote Collection capabilities."""
    
    def __init__(
        self, 
        insurance_service: InsuranceService,
        ams360_service: AMS360Service,
        agencyzoom_service: AgencyZoomService,
        *args, 
        **kwargs
    ):
        """Initialize the telephony agent.
        
        Args:
            insurance_service: The insurance service instance
            ams360_service: The AMS360 SOAP service instance
            agencyzoom_service: The AgencyZoom API service instance
        """
        super().__init__(*args, **kwargs)
        self.insurance_service = insurance_service
        self.ams360_service = ams360_service
        self.agencyzoom_service = agencyzoom_service
        
        # Initialize tool sets
        self.base_tools = BaseTools()
        self.insurance_tools = InsuranceTools(insurance_service)

    # @function_tool()
    # asyn
    
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
        first_name: str,
        last_name: str,
        date_of_birth: str,
        phone: str,
        street_address: str,
        city: str,
        state: str,
        country: str,
        zip_code: str,
        email: str,
        current_provider: str = None,
        spouse_first_name: str = None,
        spouse_last_name: str = None,
        spouse_dob: str = None,
        has_solar_panels: bool = False,
        has_pool: bool = False,
        roof_age: int = 0,
        has_pets: bool = False,
        renewal_date: str = None,
        renewal_premium: float = None
    ) -> str:
        """Collect home insurance information from the caller.
        Args:
            first_name: First name of primary insured
            last_name: Last name of primary insured
            date_of_birth: Date of birth (YYYY-MM-DD format)
            phone: Phone number
            street_address: Street address
            city: City
            state: State
            country: Country
            zip_code: ZIP or postal code
            email: Email address
            current_provider: Current insurance provider (optional)
            spouse_first_name: Spouse first name (optional)
            spouse_last_name: Spouse last name (optional)
            spouse_dob: Spouse date of birth (optional, YYYY-MM-DD format)
            has_solar_panels: Whether property has solar panels
            has_pool: Whether property has a pool
            roof_age: Age of roof in years
            has_pets: Whether household has pets
            renewal_date: Current policy renewal date (optional, YYYY-MM-DD format)
            renewal_premium: Current renewal premium amount (optional)
        """
        logger.info(f"🔧 Agent tool called: collect_home_insurance_data({first_name} {last_name})")
        # Combine first and last name for internal storage
        full_name = f"{first_name} {last_name}".strip()
        spouse_name = f"{spouse_first_name} {spouse_last_name}".strip() if spouse_first_name and spouse_last_name else None
        return self.insurance_service.collect_home_insurance(
            full_name=full_name,
            date_of_birth=date_of_birth,
            phone=phone,
            street_address=street_address,
            city=city,
            state=state,
            country=country,
            zip_code=zip_code,
            email=email,
            current_provider=current_provider,
            spouse_name=spouse_name,
            spouse_dob=spouse_dob,
            has_solar_panels=has_solar_panels,
            has_pool=has_pool,
            roof_age=roof_age,
            has_pets=has_pets,
            renewal_date=renewal_date,
            renewal_premium=renewal_premium
        )
        
    @function_tool()
    async def collect_auto_insurance_data(
        self,
        first_name: str,
        last_name: str,
        driver_dob: str,
        phone: str,
        license_number: str,
    ) -> str:
        """Collect auto insurance information from the caller.
        Args:
            driver_first_name: Driver's first name
            driver_last_name: Driver's last name
            driver_dob: Driver date of birth (YYYY-MM-DD format)
            phone: Phone number
            license_number: Driver's license number
            vin: Vehicle VIN (17 characters)
            vehicle_make: Vehicle make
            vehicle_model: Vehicle model
            coverage_type: Coverage type - "liability" or "full"
            email: Email address (optional)
            qualification: Driver qualification (optional)
            profession: Driver profession (optional)
            gpa: GPA if driver under 21 (optional)
            current_provider: Current insurance provider (optional)
            renewal_date: Current policy renewal date (optional, YYYY-MM-DD format)
            renewal_premium: Current renewal premium amount (optional)
        """
        logger.info(f"🔧 Agent tool called: collect_auto_insurance_data({driver_first_name} {driver_last_name})")
        # Combine first and last name for internal storage
        driver_name = f"{driver_first_name} {driver_last_name}".strip()
        return self.insurance_service.collect_auto_insurance(
            driver_name=driver_name,
            driver_dob=driver_dob,
            phone=phone,
            license_number=license_number,
            vin=vin,
            vehicle_make=vehicle_make,
            vehicle_model=vehicle_model,
            coverage_type=coverage_type,
            email=email,
            qualification=qualification,
            profession=profession,
            gpa=gpa,
            current_provider=current_provider,
            renewal_date=renewal_date,
            renewal_premium=renewal_premium
        )
    
    @function_tool()
    async def collect_flood_insurance_data(
        self, 
        first_name: str,
        last_name: str,
        email: str,
        phone: str,
        street_address: str,
        city: str,
        state: str,
        country: str,
        zip_code: str
    ) -> str:
        """Collect flood insurance information from the caller.
        Args:
            first_name: First name of insured
            last_name: Last name of insured
            email: Email address
            phone: Phone number
            street_address: Street address
            city: City
            state: State
            country: Country
            zip_code: ZIP or postal code
        """
        logger.info(f"🔧 Agent tool called: collect_flood_insurance_data({first_name} {last_name})")
        # Combine first and last name for internal storage
        full_name = f"{first_name} {last_name}".strip()
        return self.insurance_service.collect_flood_insurance(
            full_name=full_name,
            email=email,
            phone=phone,
            street_address=street_address,
            city=city,
            state=state,
            country=country,
            zip_code=zip_code
        )
    
    @function_tool()
    async def collect_life_insurance_data(
        self,
        first_name: str,
        last_name: str,
        date_of_birth: str,
        phone: str,
        street_address: str,
        city: str,
        state: str,
        country: str,
        zip_code: str,
        email: str = "",
        appointment_requested: bool = False,
        appointment_date: str = None,
        policy_type: str = None
    ) -> str:
        """Collect life insurance information from the caller.
        Args:
            first_name: First name of insured
            last_name: Last name of insured
            date_of_birth: Date of birth (YYYY-MM-DD format)
            phone: Phone number
            street_address: Street address
            city: City
            state: State
            country: Country
            zip_code: ZIP or postal code
            email: Email address (optional)
            appointment_requested: Whether customer wants an appointment
            appointment_date: Requested appointment date and time (optional, YYYY-MM-DD HH:MM format)
            policy_type: Type of policy - "term", "whole", "universal", "annuity", or "long_term_care" (optional)
        """
        logger.info(f"🔧 Agent tool called: collect_life_insurance_data({first_name} {last_name})")
        # Combine first and last name for internal storage
        full_name = f"{first_name} {last_name}".strip()
        return self.insurance_service.collect_life_insurance(
            full_name=full_name,
            date_of_birth=date_of_birth,
            phone=phone,
            street_address=street_address,
            city=city,
            state=state,
            country=country,
            zip_code=zip_code,
            email=email,
            appointment_requested=appointment_requested,
            appointment_date=appointment_date,
            policy_type=policy_type
        )
    
    @function_tool()
    async def collect_commercial_insurance_data(
        self,
        business_name: str,
        phone: str,
        street_address: str,
        city: str,
        state: str,
        country: str,
        zip_code: str,
        business_type: str = "General",
        email: str = "",
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
            phone: Phone number
            street_address: Street address
            city: City
            state: State
            country: Country
            zip_code: ZIP or postal code
            business_type: Type of business
            email: Email address (optional)
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
            phone=phone,
            street_address=street_address,
            city=city,
            state=state,
            country=country,
            zip_code=zip_code,
            business_type=business_type,
            email=email,
            inventory_limit=inventory_limit,
            building_coverage=building_coverage,
            building_coverage_limit=building_coverage_limit,
            current_provider=current_provider,
            renewal_date=renewal_date,
            renewal_premium=renewal_premium
        )
    
    @function_tool()
    async def submit_quote_request(self) -> str:
        """Submit the collected insurance quote request."""
        logger.info("🔧 Agent tool called: submit_quote_request()")
        return self.insurance_service.submit_quote_request()
    
    
    @function_tool()
    async def get_ams360_policy_by_number(self, policy_number: str) -> str:
        """Get policy information by policy number from AMS360.
        
        Args:
            policy_number: The policy number to search for, Please validate the policy number before calling this function.
        
        Returns:
            String message with policy information or error
        """
        logger.info(f"🔧 Agent tool called: get_ams360_policy_by_number({policy_number})")
        
        try:
            from formating.full_policy import extract_policy_fields, extract_customer_fields
            
            result = self.ams360_service.get_policy_by_number(policy_number)
            if result:
                try:
                    # Unpack the three return values: policy_details, customer_data, policy_list
                    policy_details, customer_data, policy_list = result
                    
                    # Extract policy fields using the formatting function
                    policy_info = extract_policy_fields(policy_details)
                    
                    # Format dates nicely
                    def format_date(date_str):
                        if date_str and 'T' in str(date_str):
                            return date_str.split('T')[0]
                        return date_str or 'N/A'
                    
                    # Extract customer info if available
                    customer_name = "N/A"
                    if customer_data:
                        try:
                            customer_info = extract_customer_fields(customer_data)
                            customer_name = f"{customer_info.get('FirstName', '')} {customer_info.get('LastName', '')}".strip()
                        except Exception as e:
                            logger.warning(f"Could not extract customer name: {e}")
                    
                    # Build simplified message with essential information
                    message = f"Found Policy in AMS360:\n\n"
                    message += f"Policy Number: {policy_info.get('PolicyNumber', 'N/A')}\n"
                    message += f"Policy Holder: {customer_name}\n"
                    message += f"Policy Type: {policy_info.get('PolicyTypeOfBusiness', 'N/A')}\n"
                    message += f"Line of Business: {policy_info.get('LineDescription', 'N/A')}\n"
                    message += f"Effective Date: {format_date(policy_info.get('EffectiveDate'))}\n"
                    message += f"Expiration Date: {format_date(policy_info.get('ExpirationDate'))}\n"
                    message += f"Full Term Premium: ${policy_info.get('FullTermPremium', 'N/A')}\n"
                    message += f"Bill Method: {policy_info.get('BillMethod', 'N/A')}\n"
                    
                    # Add latest transaction info if available
                    if policy_info.get('LatestTransactionType'):
                        message += f"\nLatest Transaction:\n"
                        message += f"   Type: {policy_info.get('LatestTransactionType', 'N/A')}\n"
                        message += f"   Date: {format_date(policy_info.get('LatestTransactionDate'))}\n"
                        message += f"   Premium: ${policy_info.get('LatestPremium', 'N/A')}\n"
                    
                    return message
                    
                except Exception as e:
                    logger.warning(f"Error formatting policy details: {e}")
                    return f"Found policy information in AMS360 for policy number {policy_number}."
            else:
                return f"No policy found in AMS360 with policy number {policy_number}."
        
        except Exception as e:
            logger.error(f"Error getting AMS360 policy by number: {e}")
            return f"Error retrieving policy: {str(e)}"
    
    
    # AgencyZoom Integration Tools
    @function_tool()
    async def create_agencyzoom_lead(
        self, 
        first_name: str, 
        last_name: str, 
        email: str, 
        phone: str, 
        insurance_type: str,
        streetAddress: str = "",
        city: str = "",
        state: str = "",
        country: str = "",
        zip_code: str = "",
        notes: str = "",
        birthday: str = "",
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
            streetAddress: Street address (optional)
            city: City (optional)
            state: State (optional)
            country: Country (optional)
            zip_code: ZIP or postal code (optional)
            notes: Additional notes about the lead (optional)
            birthday: Date of birth for life/personal insurance (optional)
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
        
        # Build full address if components provided
        address_parts = [streetAddress, city, state, zip_code, country]
        full_address = ", ".join([part for part in address_parts if part])
        if full_address:
            lead_data["address"] = full_address
            
        # Add optional fields if provided
        if birthday:
            lead_data["date_of_birth"] = birthday
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
    async def search_agencyzoom_contact_by_phone(self, phone: str) -> str:
        """Search for a contact in AgencyZoom by phone number.
        
        Args:
            phone: Phone number to search for
        
        Returns:
            String message with search results or error
        """
        logger.info(f"🔧 Agent tool called: search_agencyzoom_contact_by_phone({phone})")
        
        try:
            result = self.agencyzoom_service.search_contact_by_phone(phone)
            if result and result.get('contacts'):
                count = len(result['contacts'])
                return f"Found {count} contact(s) in AgencyZoom with phone number {phone}."
            else:
                return f"No contact found in AgencyZoom with phone number {phone}."
        except Exception as e:
            logger.error(f"Error searching AgencyZoom by phone: {e}")
            return f"Error searching AgencyZoom: {str(e)}"
    
    @function_tool()
    async def search_agencyzoom_contact_by_email(self, email: str) -> str:
        """Search for a contact in AgencyZoom by email address.
        
        Args:
            email: Email address to search for
        
        Returns:
            String message with search results or error
        """
        logger.info(f"🔧 Agent tool called: search_agencyzoom_contact_by_email({email})")
        
        try:
            result = self.agencyzoom_service.search_contact_by_email(email)
            if result and result.get('contacts'):
                count = len(result['contacts'])
                return f"Found {count} contact(s) in AgencyZoom with email {email}."
            else:
                return f"No contact found in AgencyZoom with email {email}."
        except Exception as e:
            logger.error(f"Error searching AgencyZoom by email: {e}")
            return f"Error searching AgencyZoom: {str(e)}"
    
    @function_tool()
    async def add_note_to_agencyzoom_contact(self, contact_id: str, note: str) -> str:
        """Add a note to an existing contact in AgencyZoom.
        
        Args:
            contact_id: The AgencyZoom contact ID
            note: The note text to add
        
        Returns:
            String message confirming note addition or error
        """
        logger.info(f"🔧 Agent tool called: add_note_to_agencyzoom_contact({contact_id})")
        
        try:
            result = self.agencyzoom_service.add_note_to_contact(contact_id, note)
            if result:
                return f"Successfully added note to contact {contact_id} in AgencyZoom."
            else:
                return "Failed to add note to AgencyZoom contact. Please check the logs."
        except Exception as e:
            logger.error(f"Error adding note to AgencyZoom contact: {e}")
            return f"Error adding note: {str(e)}"
    
    
    async def before_llm_inference(self, ctx: RunContext):
        """
        Hook that runs BEFORE the LLM is called. We inject RAG context here
        so the agent can speak immediately without tool call delays.
        
        This is a lifecycle hook, not a function tool - it runs automatically.
        """
        # Get the chat context
        chat_ctx = ctx.chat_context
        if not chat_ctx or not chat_ctx.messages:
            return
        
        # Get the user's last message
        last_message = chat_ctx.messages[-1]
        if last_message.role != "user":
            return
        
        user_query = last_message.content
        logger.info(f"🔍 Proactive RAG search for: {user_query}")
        
        if not self.rag_service:
            logger.warning("RAG service not available")
            return
        
        try:
            # Quick RAG search with timeout to prevent blocking
            search_results = await asyncio.wait_for(
                asyncio.to_thread(
                    self.rag_service.retrieval_based_search,
                    query=user_query,
                    collections=["island"],  # Search all collections
                    top_k=1  # Get top result only for speed
                ),
                timeout=0.5  # 1 second max to keep conversation flowing
            )
            
            if search_results and len(search_results) > 0:
                context = search_results[0].get('text', '').strip()
                score = search_results[0].get('score', 0)
                source = search_results[0].get('source', 'unknown')
                
                if context:
                    # Inject context into the chat as a system message
                    chat_ctx.append(
                        role="system",
                        text=f"Relevant context from knowledge base (relevance: {score:.2f}):\n{context}\n\nSource: {source}"
                    )
                    logger.info(f"✓ RAG context injected (score: {score:.2f})")
            else:
                logger.info("No RAG results found - using general knowledge")
                
        except asyncio.TimeoutError:
            logger.warning("⚠️ RAG search timed out - continuing without context")
        except Exception as e:
            logger.error(f"RAG search error: {e}")
            # Don't fail the conversation if RAG fails


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
                property_addr = insurance_data.get("property", {}).get("address", {})
                lead_data["streetAddress"] = property_addr.get('streetAddress', '')
                lead_data["city"] = property_addr.get('city', '')
                lead_data["state"] = property_addr.get('state', '')
                lead_data["country"] = property_addr.get('country', '')
                lead_data["zip"] = property_addr.get('zip_code', '')
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
                home_addr = insurance_data.get("home_address", {})
                lead_data["streetAddress"] = home_addr.get('streetAddress', '')
                lead_data["city"] = home_addr.get('city', '')
                lead_data["state"] = home_addr.get('state', '')
                lead_data["country"] = home_addr.get('country', '')
                lead_data["zip"] = home_addr.get('zip_code', '')
                
            elif insurance_type == "life":
                life_addr = insurance_data.get("address", {})
                lead_data["streetAddress"] = life_addr.get('streetAddress', '')
                lead_data["city"] = life_addr.get('city', '')
                lead_data["state"] = life_addr.get('state', '')
                lead_data["country"] = life_addr.get('country', '')
                lead_data["zip"] = life_addr.get('zip_code', '')
                lead_data["appointment_requested"] = insurance_data.get("appointment_requested", False)
                lead_data["appointment_date"] = insurance_data.get("appointment_date", "")
                lead_data["policy_type"] = insurance_data.get("policy_type", "")
                lead_data["date_of_birth"] = insurance_data.get("insured", {}).get("date_of_birth", "")
                
            elif insurance_type == "commercial":
                lead_data["business_name"] = insurance_data.get("business", {}).get("name", "")
                lead_data["business_type"] = insurance_data.get("business", {}).get("type", "")
                business_addr = insurance_data.get("business", {}).get("address", {})
                lead_data["streetAddress"] = business_addr.get('streetAddress', '')
                lead_data["city"] = business_addr.get('city', '')
                lead_data["state"] = business_addr.get('state', '')
                lead_data["country"] = business_addr.get('country', '')
                lead_data["zip"] = business_addr.get('zip_code', '')
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


async def entrypoint(ctx: JobContext):
    """Main entry point for the telephony voice agent."""
    await ctx.connect()
    
    # Wait for participant (caller) to join
    participant = await ctx.wait_for_participant()
    logger.info(f"Phone call connected from participant: {participant.identity}")
    
    # Initialize all services
    ams360_service = AMS360Service()
    agencyzoom_service = AgencyZoomService()
    insurance_service = InsuranceService(agencyzoom_service=agencyzoom_service)
    
    # Build comprehensive instructions with knowledge base
#     base_instructions = AGENT_SYSTEM_INSTRUCTIONS
    
    instructions="""You are an AI insurance assistant for Inshora Group.

INTRODUCTION: "Hello, this is the AI insurance assistant from Inshora Group. How can I help you today?"

PERSONALITY:
- Professional, warm, and concise
- Speak clearly at moderate pace
- Patient but efficient
- Guide naturally through information collection

CORE WORKFLOWS:

1. EXISTING POLICY LOOKUP:
   - Get full name and policy number
   - Confirm policy number by repeating it back
   - Say "Give me a moment while I pull up your policy"
   - Call get_policy tool ONCE
   - Verify name matches; if not, ask customer to spell registered name
   - Share policy details

2. NEW INSURANCE QUOTE:
   - Ask: ADD new or UPDATE existing?
   - Identify insurance type: home, auto, flood, life, commercial
   - Call set_action tool first
   - Collect required info using appropriate collect tool
   - Say "Let me submit your request" before calling submit tools
   - Call BOTH submit_quote AND save_to_crm
   - Confirm submission

CRITICAL RULES:
- Call get_policy only ONCE per policy number
- Always call set_action before collecting data
- Always call BOTH submit_quote AND save_to_crm for new quotes
- Inform user before submitting data
- Keep responses brief and conversational
- Ask 1-2 questions at a time, not all at once

ESCALATION: Transfer to human if customer mentions: lawsuit, claim denied, urgent cancellation, expresses anger, or explicitly requests human agent.

DATES: Request format YYYY-MM-DD (e.g., "1980-05-15")
VIN: Must be exactly 17 characters"""

    # Number of characters in the instructions
    num_characters = len(instructions)
    logger.info(f"Number of characters in the instructions: {num_characters}")
    
    # Number of tokens in the instructions
    num_tokens = len(instructions.split())
    logger.info(f"Number of tokens in the instructions: {num_tokens}")
    
    # Initialize the insurance receptionist agent with function tools
    agent = TelephonyAgent(
        insurance_service=insurance_service,
        ams360_service=ams360_service,
        agencyzoom_service=agencyzoom_service,
        instructions=instructions
    )
    
    logger.info("=" * 60)
    logger.info("Telephony Agent Initialized")
    logger.info(f"Knowledge Base: Loaded ({len(INSHORA_KNOWLEDGE_BASE)} characters)")
    logger.info("Available Tools:")
    logger.info("  - Insurance Data Collection (7 tools)")
    logger.info("  - AMS360 Policy Lookup (4 tools)")
    logger.info("  - AgencyZoom CRM Integration (5 tools)")
    logger.info(f"Total Tools: 16")
    logger.info(f"Instructions Length: {len(instructions)} characters")
    logger.info("=" * 60)
    
    # Configure the voice processing pipeline optimized for telephony
    session = AgentSession(
        # Voice Activity Detection
        # vad=silero.VAD.load(),
        # Create the realtime model
        llm = openai.realtime.RealtimeModel(
            api_key=os.getenv("OPENAI_API_KEY"),
            voice="cedar",
            model="gpt-realtime",
            temperature=0.8,
            turn_detection=TurnDetection(
                type="server_vad",
                silence_duration_ms=800,  # Reduced for lower latency - more responsive
                prefix_padding_ms=300,     # Reduced for lower latency
                threshold=0.5,  # Lower threshold for more responsive detection
            ),
            max_session_duration=1800,
        ))
        # # Speech-to-Text - Deepgram Nova-3
        # stt=deepgram.STT(
        #     model=config.stt.model,
        #     language=config.stt.language,
        #     interim_results=config.stt.interim_results,
        #     punctuate=config.stt.punctuate,
        #     smart_format=config.stt.smart_format,
        #     filler_words=config.stt.filler_words,
        #     endpointing_ms=config.stt.endpointing_ms,
        #     sample_rate=config.stt.sample_rate
        # ),
        
        # # Large Language Model - GPT-4o-mini
        # llm=openai.LLM(
        #     model=config.llm.model,
        #     temperature=config.llm.temperature
        # ),
        
        # Text-to-Speech - Cartesia Sonic-2
        # tts=cartesia.TTS(
        #     model=config.tts.model,
        #     voice=config.tts.voice,
        #     language=config.tts.language,
        #     speed=config.tts.speed,
        #     sample_rate=config.tts.sample_rate
        # )
        # tts = elevenlabs.TTS(
        #         base_url="https://api.eu.residency.elevenlabs.io/v1",
        #         voice_id="21m00Tcm4TlvDq8ikWAM",
        #         language="en",
        #         model="eleven_flash_v2_5"
        #     ))
    
    logger.info("AgentSession configured successfully")
    
    # Start the agent session - tools are automatically discovered from agent methods
    await session.start(agent=agent, room=ctx.room)
    
    logger.info("Agent session started - function tools are automatically available to the LLM")
    
    # Generate personalized greeting based on time of day
    hour = datetime.now().hour
    if hour < 12:
        time_greeting = "Good morning"
    elif hour < 18:
        time_greeting = "Good afternoon"
    else:
        time_greeting = "Good evening"
    
    await session.generate_reply(
        instructions=get_greeting_prompt(time_greeting)
    )


if __name__ == "__main__":
    # Configure logging for better debugging - log to both console and file
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Create file handler
    file_handler = logging.FileHandler('agent.log', mode='a', encoding='utf-8')
    file_handler.setLevel(getattr(logging, config.log_level))
    file_handler.setFormatter(logging.Formatter(log_format))
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, config.log_level))
    console_handler.setFormatter(logging.Formatter(log_format))
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, config.log_level),
        format=log_format,
        handlers=[file_handler, console_handler]
    )
    
    logger.info("Starting telephony agent - logs will be saved to agent.log")
    
    # Run the agent with the name that matches your dispatch rule
    cli.run_app(WorkerOptions(
        entrypoint_fnc=entrypoint,
        agent_name="inshora-agent"  # This must match your dispatch rule
    ))
