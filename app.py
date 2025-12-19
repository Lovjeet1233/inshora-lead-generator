"""
Unified API Application
Combines Chatbot, SMS, and Email services
"""

import logging
import os
import json
from datetime import datetime
from typing import Dict, Optional, List
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import openai
from outboundService.services.call_service import make_outbound_call

from outboundService.common.update_config import update_config_async
from models.model import OutboundCallRequest, StatusResponse
from services.insurance_service import InsuranceService
from services.ams360 import AMS360Service
from services.agencyzoom import AgencyZoomService
from config import AGENT_SYSTEM_INSTRUCTIONS, CHATBOT_SYSTEM_INSTRUCTIONS, get_knowledge_base

# Import routers
from routers import sms, email

load_dotenv()
logger = logging.getLogger("unified-api")

# Load Knowledge Base for chatbot
KNOWLEDGE_BASE = get_knowledge_base()
logger.info("Inshora Knowledge Base loaded successfully for chatbot")


# ===========================
# UTILITY FUNCTIONS FOR OUTBOUND CALLS
# ===========================

def log_info(message: str):
    """Log info message."""
    logger.info(message)

def log_error(message: str):
    """Log error message."""
    logger.error(message)

def log_exception(message: str):
    """Log exception message."""
    logger.exception(message)

def format_phone_number(phone: str) -> str:
    """
    Format phone number to E.164 format.
    Removes spaces, dashes, parentheses, and ensures + prefix.
    """
    # Remove common separators
    cleaned = phone.replace(" ", "").replace("-", "").replace("(", "").replace(")", "").replace(".", "")
    
    # Ensure + prefix
    if not cleaned.startswith("+"):
        # Assume US number if no country code
        if len(cleaned) == 10:
            cleaned = "+1" + cleaned
        elif len(cleaned) == 11 and cleaned.startswith("1"):
            cleaned = "+" + cleaned
        else:
            cleaned = "+" + cleaned
    
    return cleaned

def validate_phone_number(phone: str) -> bool:
    """
    Validate phone number format.
    Must start with + and contain only digits after.
    """
    if not phone.startswith("+"):
        return False
    
    # Check rest is digits
    digits = phone[1:]
    if not digits.isdigit():
        return False
    
    # Must be between 10-15 digits (E.164 standard)
    if len(digits) < 10 or len(digits) > 15:
        return False
    
    return True


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('api.log', mode='a', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# Initialize FastAPI app
app = FastAPI(
    title="Unified Insurance API",
    version="1.0.0",
    description="Comprehensive API for Insurance Chatbot, SMS, and Email services"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(sms.router)
app.include_router(email.router)

# Initialize OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")

# In-memory conversation storage (thread_id -> messages list)
# In production, you'd use a database like PostgreSQL, MongoDB, or Redis
conversation_threads: Dict[str, List[Dict]] = {}

# Store service instances per thread (for maintaining session state)
thread_services: Dict[str, Dict] = {}

# Store detailed policy information per thread (for on-demand retrieval)
thread_policy_details: Dict[str, Dict] = {}

# Store escalation state per thread (for tracking handover status)
thread_escalation_state: Dict[str, Dict] = {}


# ===========================
# CHATBOT MODELS
# ===========================

class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    query: str
    thread_id: str
    prompt: Optional[str] = None  # Custom system prompt (if None, uses default CHATBOT_SYSTEM_INSTRUCTIONS)
    escalation_condition: Optional[str] = None  # Condition to trigger handover to human
    reset_escalation: bool = False  # Set to True to reset escalation and continue with bot


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""
    response: str
    thread_id: str
    timestamp: str
    requires_handover: bool = False  # True if escalation condition is met
    handover_reason: Optional[str] = None  # Reason for handover
    escalation_active: bool = False  # True if currently in escalated state
    escalation_reset: bool = False  # True if escalation was just reset


# ===========================
# CHATBOT SERVICE FUNCTIONS
# ===========================

def get_or_create_thread_services(thread_id: str) -> Dict:
    """Get or create service instances for a thread."""
    if thread_id not in thread_services:
        # Initialize services for this thread
        ams360_service = AMS360Service()
        agencyzoom_service = AgencyZoomService()
        insurance_service = InsuranceService(agencyzoom_service=agencyzoom_service)
        
        thread_services[thread_id] = {
            "insurance": insurance_service,
            "ams360": ams360_service,
            "agencyzoom": agencyzoom_service
        }
        logger.info(f"Created new service instances for thread: {thread_id}")
    
    return thread_services[thread_id]


def get_or_create_thread(thread_id: str, custom_prompt: Optional[str] = None) -> List[Dict]:
    """
    Get or create a conversation thread.
    
    Args:
        thread_id: Unique identifier for the conversation thread
        custom_prompt: Optional custom system prompt. If None, uses default CHATBOT_SYSTEM_INSTRUCTIONS
    
    Returns:
        List of conversation messages for the thread
    """
    # Determine which prompt to use
    system_prompt = custom_prompt if custom_prompt is not None else CHATBOT_SYSTEM_INSTRUCTIONS
    
    if thread_id not in conversation_threads:
        # Initialize with system message
        conversation_threads[thread_id] = [
            {
                "role": "system",
                "content": system_prompt
            }
        ]
        prompt_type = "custom prompt" if custom_prompt else "default Inshora Knowledge Base"
        logger.info(f"Created new conversation thread with {prompt_type}: {thread_id}")
    else:
        # Thread exists - update system message if custom prompt is provided
        if custom_prompt is not None:
            conversation_threads[thread_id][0] = {
                "role": "system",
                "content": system_prompt
            }
            logger.info(f"Updated system prompt for existing thread: {thread_id}")
    
    return conversation_threads[thread_id]


def get_available_tools() -> List[Dict]:
    """Define available function tools for the chatbot."""
    return [
        {
            "type": "function",
            "function": {
                "name": "set_user_action",
                "description": "Set the user action type (add/update) and insurance type.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action_type": {
                            "type": "string",
                            "enum": ["add", "update"],
                            "description": "Either 'add' for new insurance or 'update' for existing policy"
                        },
                        "insurance_type": {
                            "type": "string",
                            "enum": ["home", "auto", "flood", "life", "commercial"],
                            "description": "Type of insurance"
                        }
                    },
                    "required": ["action_type", "insurance_type"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "collect_home_insurance_data",
                "description": "Collect home insurance information from the user.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "full_name": {"type": "string", "description": "Full name of primary insured"},
                        "date_of_birth": {"type": "string", "description": "Date of birth (YYYY-MM-DD format)"},
                        "property_address": {"type": "string", "description": "Property address"},
                        "phone": {"type": "string", "description": "Phone number"},
                        "email": {"type": "string", "description": "Email address"},
                        "spouse_name": {"type": "string", "description": "Spouse name (optional)"},
                        "spouse_dob": {"type": "string", "description": "Spouse date of birth (YYYY-MM-DD format, optional)"},
                        "has_solar_panels": {"type": "boolean", "description": "Whether property has solar panels"},
                        "has_pool": {"type": "boolean", "description": "Whether property has a pool"},
                        "roof_age": {"type": "integer", "description": "Age of roof in years"},
                        "has_pets": {"type": "boolean", "description": "Whether household has pets"},
                        "current_provider": {"type": "string", "description": "Current insurance provider (optional)"},
                        "renewal_date": {"type": "string", "description": "Current policy renewal date (YYYY-MM-DD format, optional)"},
                        "renewal_premium": {"type": "number", "description": "Current renewal premium amount (optional)"}
                    },
                    "required": ["full_name", "date_of_birth", "property_address", "phone", "email"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "collect_auto_insurance_data",
                "description": "Collect auto insurance information from the user.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "driver_name": {"type": "string", "description": "Full name of driver"},
                        "driver_dob": {"type": "string", "description": "Driver date of birth (YYYY-MM-DD format)"},
                        "license_number": {"type": "string", "description": "Driver's license number"},
                        "qualification": {"type": "string", "description": "Driver qualification"},
                        "profession": {"type": "string", "description": "Driver profession"},
                        "vin": {"type": "string", "description": "Vehicle VIN (17 characters)"},
                        "vehicle_make": {"type": "string", "description": "Vehicle make"},
                        "vehicle_model": {"type": "string", "description": "Vehicle model"},
                        "phone": {"type": "string", "description": "Phone number"},
                        "email": {"type": "string", "description": "Email address"},
                        "gpa": {"type": "number", "description": "GPA if driver under 21 (optional)"},
                        "coverage_type": {"type": "string", "description": "Coverage type - 'liability' or 'full'"},
                        "current_provider": {"type": "string", "description": "Current insurance provider (optional)"},
                        "renewal_date": {"type": "string", "description": "Current policy renewal date (YYYY-MM-DD format, optional)"},
                        "renewal_premium": {"type": "number", "description": "Current renewal premium amount (optional)"}
                    },
                    "required": ["driver_name", "driver_dob", "license_number", "qualification", "profession", "vin", "vehicle_make", "vehicle_model", "phone", "email"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "collect_flood_insurance_data",
                "description": "Collect flood insurance information from the user.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "full_name": {"type": "string", "description": "Full name of insured"},
                        "home_address": {"type": "string", "description": "Home address for flood insurance"},
                        "email": {"type": "string", "description": "Email address"}
                    },
                    "required": ["full_name", "home_address", "email"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "collect_life_insurance_data",
                "description": "Collect life insurance information from the user.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "full_name": {"type": "string", "description": "Full name of insured"},
                        "date_of_birth": {"type": "string", "description": "Date of birth (YYYY-MM-DD format)"},
                        "appointment_requested": {"type": "boolean", "description": "Whether customer wants an appointment"},
                        "phone": {"type": "string", "description": "Phone number"},
                        "email": {"type": "string", "description": "Email address"},
                        "appointment_date": {"type": "string", "description": "Requested appointment date and time (YYYY-MM-DD HH:MM format, optional)"},
                        "policy_type": {"type": "string", "description": "Type of policy - 'term', 'whole', 'universal', 'annuity', or 'long_term_care' (optional)"}
                    },
                    "required": ["full_name", "date_of_birth", "appointment_requested", "phone", "email"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "collect_commercial_insurance_data",
                "description": "Collect commercial insurance information from the user.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "business_name": {"type": "string", "description": "Name of the business"},
                        "business_type": {"type": "string", "description": "Type of business"},
                        "business_address": {"type": "string", "description": "Business address"},
                        "phone": {"type": "string", "description": "Phone number"},
                        "email": {"type": "string", "description": "Email address"},
                        "inventory_limit": {"type": "number", "description": "Inventory coverage limit (optional)"},
                        "building_coverage": {"type": "boolean", "description": "Whether building coverage is needed"},
                        "building_coverage_limit": {"type": "number", "description": "Building coverage limit (optional)"},
                        "current_provider": {"type": "string", "description": "Current insurance provider (optional)"},
                        "renewal_date": {"type": "string", "description": "Current policy renewal date (YYYY-MM-DD format, optional)"},
                        "renewal_premium": {"type": "number", "description": "Current renewal premium amount (optional)"}
                    },
                    "required": ["business_name", "business_type", "business_address", "phone", "email"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "submit_quote_request",
                "description": "Submit the collected insurance quote request.",
                "parameters": {
                    "type": "object",
                    "properties": {}
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_policy_by_number",
                "description": "Get policy information or lookup for existing policy by policy number from AMS360. Returns basic/major information.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "policy_number": {"type": "string", "description": "The policy number to search for"}
                    },
                    "required": ["policy_number"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_detailed_policy_info",
                "description": "Get additional detailed information about a previously looked up policy (transactions, customer contact details, etc.).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "policy_number": {"type": "string", "description": "The policy number to get details for"}
                    },
                    "required": ["policy_number"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "create_agencyzoom_lead",
                "description": "Create a new lead in AgencyZoom with detailed information.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "first_name": {"type": "string"},
                        "last_name": {"type": "string"},
                        "email": {"type": "string"},
                        "phone": {"type": "string"},
                        "insurance_type": {"type": "string"},
                        "notes": {"type": "string"},
                        "address": {"type": "string"},
                        "date_of_birth": {"type": "string"},
                        "current_provider": {"type": "string"},
                        "vehicle_info": {"type": "string"},
                        "property_info": {"type": "string"},
                        "business_name": {"type": "string"},
                        "appointment_requested": {"type": "boolean"}
                    },
                    "required": ["first_name", "last_name", "email", "phone", "insurance_type"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "submit_collected_data_to_agencyzoom",
                "description": "Submit all collected insurance data to AgencyZoom as a comprehensive lead.",
                "parameters": {
                    "type": "object",
                    "properties": {}
                }
            }
        }
    ]


def execute_function_call(function_name: str, arguments: Dict, thread_id: str) -> str:
    """Execute a function call and return the result."""
    services = get_or_create_thread_services(thread_id)
    insurance_service = services["insurance"]
    ams360_service = services["ams360"]
    agencyzoom_service = services["agencyzoom"]
    
    logger.info(f"🔧 Executing function: {function_name} with args: {arguments}")
    
    try:
        # Insurance Service Functions
        if function_name == "set_user_action":
            return insurance_service.set_user_action(
                arguments.get("action_type"),
                arguments.get("insurance_type")
            )
        
        elif function_name == "collect_home_insurance_data":
            return insurance_service.collect_home_insurance(
                full_name=arguments.get("full_name"),
                date_of_birth=arguments.get("date_of_birth"),
                spouse_name=arguments.get("spouse_name"),
                spouse_dob=arguments.get("spouse_dob"),
                property_address=arguments.get("property_address"),
                has_solar_panels=arguments.get("has_solar_panels", False),
                has_pool=arguments.get("has_pool", False),
                roof_age=arguments.get("roof_age", 0),
                has_pets=arguments.get("has_pets", False),
                current_provider=arguments.get("current_provider"),
                renewal_date=arguments.get("renewal_date"),
                renewal_premium=arguments.get("renewal_premium"),
                phone=arguments.get("phone"),
                email=arguments.get("email")
            )
        
        elif function_name == "collect_auto_insurance_data":
            return insurance_service.collect_auto_insurance(
                driver_name=arguments.get("driver_name"),
                driver_dob=arguments.get("driver_dob"),
                license_number=arguments.get("license_number"),
                qualification=arguments.get("qualification"),
                profession=arguments.get("profession"),
                gpa=arguments.get("gpa"),
                vin=arguments.get("vin"),
                vehicle_make=arguments.get("vehicle_make"),
                vehicle_model=arguments.get("vehicle_model"),
                coverage_type=arguments.get("coverage_type", "full"),
                current_provider=arguments.get("current_provider"),
                renewal_date=arguments.get("renewal_date"),
                renewal_premium=arguments.get("renewal_premium"),
                phone=arguments.get("phone"),
                email=arguments.get("email")
            )
        
        elif function_name == "collect_flood_insurance_data":
            return insurance_service.collect_flood_insurance(
                arguments.get("full_name"),
                arguments.get("home_address"),
                arguments.get("email")
            )
        
        elif function_name == "collect_life_insurance_data":
            return insurance_service.collect_life_insurance(
                full_name=arguments.get("full_name"),
                date_of_birth=arguments.get("date_of_birth"),
                appointment_requested=arguments.get("appointment_requested"),
                appointment_date=arguments.get("appointment_date"),
                phone=arguments.get("phone"),
                email=arguments.get("email"),
                policy_type=arguments.get("policy_type")
            )
        
        elif function_name == "collect_commercial_insurance_data":
            return insurance_service.collect_commercial_insurance(
                business_name=arguments.get("business_name"),
                business_type=arguments.get("business_type"),
                business_address=arguments.get("business_address"),
                inventory_limit=arguments.get("inventory_limit"),
                building_coverage=arguments.get("building_coverage", False),
                building_coverage_limit=arguments.get("building_coverage_limit"),
                current_provider=arguments.get("current_provider"),
                renewal_date=arguments.get("renewal_date"),
                renewal_premium=arguments.get("renewal_premium"),
                phone=arguments.get("phone"),
                email=arguments.get("email")
            )
        
        elif function_name == "submit_quote_request":
            return insurance_service.submit_quote_request()
        
        # AMS360 Functions
        elif function_name == "get_policy_by_number":
            from formating.full_policy import extract_policy_fields, extract_customer_fields
            
            result, customer_data, policy_id = ams360_service.get_policy_by_number(arguments.get("policy_number"))
            if result:
                try:
                    # Extract policy fields using the formatting function
                    policy_info = extract_policy_fields(result)
                    
                    # Format dates nicely
                    def format_date(date_str):
                        if date_str and 'T' in str(date_str):
                            return date_str.split('T')[0]
                        return date_str or 'N/A'
                    
                    # Store full details for later retrieval
                    if thread_id not in thread_policy_details:
                        thread_policy_details[thread_id] = {}
                    
                    thread_policy_details[thread_id][arguments.get("policy_number")] = {
                        "policy_info": policy_info,
                        "customer_info": policy_id,
                        "format_date": format_date
                    }
                    
                    # Extract customer info if available
                    customer_name = "N/A"
                    if customer_data:
                        try:
                            customer_info = extract_customer_fields(policy_id)
                            customer_name = f"{customer_info.get('FirstName', '')} {customer_info.get('LastName', '')}".strip()
                        except Exception as e:
                            logger.warning(f"Could not extract customer name: {e}")
                    
                    # Build simplified message with ONLY major/essential information
                    message = f"✓ Found Policy in AMS360:\n\n"
                    message += f"📋 Policy Number: {policy_info.get('PolicyNumber', 'N/A')}\n"
                    message += f"👤 Customer: {customer_name}\n"
                    message += f"💼 Policy Type: {policy_info.get('PolicyTypeOfBusiness', 'N/A')}\n"
                    message += f"📅 Effective: {format_date(policy_info.get('EffectiveDate'))}\n"
                    message += f"📅 Expires: {format_date(policy_info.get('ExpirationDate'))}\n"
                    message += f"💰 Premium: ${policy_info.get('FullTermPremium', 'N/A')}\n"
                    message += f"\n💡 Ask me if you need more details (transactions, contact info, etc.)"
                    
                    return message
                    
                except Exception as e:
                    logger.warning(f"Error formatting policy details: {e}")
                    return f"Found policy information in AMS360 for policy number {arguments.get('policy_number')}."
            else:
                return f"❌ No policy found in AMS360 with policy number {arguments.get('policy_number')}."
        
        elif function_name == "get_detailed_policy_info":
            policy_number = arguments.get("policy_number")
            
            # Check if we have stored details for this policy
            if thread_id not in thread_policy_details or policy_number not in thread_policy_details[thread_id]:
                return f"No cached details found for policy {policy_number}. Please look up the policy first using get_policy_by_number."
            
            stored_data = thread_policy_details[thread_id][policy_number]
            policy_info = stored_data["policy_info"]
            customer_data = stored_data["customer_data"]
            format_date = stored_data["format_date"]
            
            from formating.full_policy import extract_customer_fields
            
            # Build detailed message
            message = f"📋 Detailed Information for Policy {policy_number}:\n\n"
            
            # Additional policy details
            message += f"━━━━━━━━━━━━━━━━━━━━━━\n"
            message += f"📊 POLICY DETAILS:\n"
            message += f"   Line of Business: {policy_info.get('LineDescription', 'N/A')}\n"
            message += f"   Bill Method: {policy_info.get('BillMethod', 'N/A')}\n"
            message += f"   Status: {policy_info.get('PolicyStatus', 'N/A')}\n"
            
            # Latest transaction info if available
            if policy_info.get('LatestTransactionType'):
                message += f"\n📝 LATEST TRANSACTION:\n"
                message += f"   Type: {policy_info.get('LatestTransactionType', 'N/A')}\n"
                message += f"   Date: {format_date(policy_info.get('LatestTransactionDate'))}\n"
                message += f"   Premium: ${policy_info.get('LatestPremium', 'N/A')}\n"
            
            # Customer contact info if available
            if customer_data:
                try:
                    customer_info = extract_customer_fields(customer_data)
                    message += f"\n👤 CUSTOMER CONTACT INFO:\n"
                    message += f"   Name: {customer_info.get('FirstName', '')} {customer_info.get('LastName', '')}\n"
                    message += f"   Customer ID: {customer_info.get('CustomerId', 'N/A')}\n"
                    
                    # Add contact info if available
                    if customer_info.get('Email'):
                        message += f"   Email: {customer_info.get('Email')}\n"
                    if customer_info.get('CellPhone'):
                        message += f"   Phone: {customer_info.get('CellAreaCode', '')}{customer_info.get('CellPhone', '')}\n"
                    if customer_info.get('City') and customer_info.get('State'):
                        message += f"   Location: {customer_info.get('City')}, {customer_info.get('State')}\n"
                except Exception as e:
                    logger.warning(f"Could not extract customer details: {e}")
            
            return message
        
        elif function_name == "get_ams360_customer_policies":
            from formating.full_policy import extract_policy_list
            
            result = ams360_service.get_customer_policies(arguments.get("customer_id"))
            if result:
                try:
                    # Extract policy list using the formatting function
                    policy_list = extract_policy_list(result)
                    
                    if not policy_list:
                        return f"No policies found for customer {arguments.get('customer_id')} in AMS360."
                    
                    # Format dates nicely
                    def format_date(date_str):
                        if date_str and 'T' in str(date_str):
                            return date_str.split('T')[0]
                        return date_str or 'N/A'
                    
                    # Build user-friendly message
                    message = f"✓ Found {len(policy_list)} Policy(ies) for Customer ID: {arguments.get('customer_id')}\n\n"
                    
                    for idx, policy in enumerate(policy_list, 1):
                        message += f"━━━━━━━━━━━━━━━━━━━━━━\n"
                        message += f"Policy #{idx}:\n"
                        message += f"📋 Policy Number: {policy.get('PolicyNumber', 'N/A')}\n"
                        message += f"💼 Type: {policy.get('PolicyTypeOfBusiness', 'N/A')}\n"
                        message += f"📊 Status: {policy.get('PolicyStatus', 'N/A')}\n"
                        message += f"📅 Effective: {format_date(policy.get('PolicyEffectiveDate'))}\n"
                        message += f"📅 Expiration: {format_date(policy.get('PolicyExpirationDate'))}\n"
                        message += f"🏢 Company: {policy.get('WritingCompanyCode', 'N/A')}\n"
                        message += "\n"
                    
                    return message
                    
                except Exception as e:
                    logger.warning(f"Error formatting policy list: {e}")
                    return f"Retrieved policies for customer {arguments.get('customer_id')} from AMS360 successfully."
            else:
                return f"❌ No policies found for customer {arguments.get('customer_id')} in AMS360."
        
        # AgencyZoom Functions
        elif function_name == "create_agencyzoom_lead":
            lead_data = {
                "first_name": arguments.get("first_name"),
                "last_name": arguments.get("last_name"),
                "email": arguments.get("email"),
                "phone": arguments.get("phone"),
                "insurance_type": arguments.get("insurance_type"),
                "notes": arguments.get("notes", ""),
                "source": "AI Chatbot"
            }
            
            # Add optional fields
            optional_fields = ["address", "date_of_birth", "current_provider", "vehicle_info", 
                             "property_info", "business_name", "appointment_requested"]
            for field in optional_fields:
                if arguments.get(field):
                    lead_data[field] = arguments.get(field)
            
            result = agencyzoom_service.create_lead(lead_data)
            if result:
                return f"Successfully created lead in AgencyZoom for {arguments.get('first_name')} {arguments.get('last_name')}."
            else:
                return "Failed to create lead in AgencyZoom. Please check the logs for details."
        
        elif function_name == "search_agencyzoom_contact_by_phone":
            result = agencyzoom_service.search_contact_by_phone(arguments.get("phone"))
            if result and result.get('contacts'):
                count = len(result['contacts'])
                return f"Found {count} contact(s) in AgencyZoom with phone number {arguments.get('phone')}."
            else:
                return f"No contact found in AgencyZoom with phone number {arguments.get('phone')}."
        
        elif function_name == "search_agencyzoom_contact_by_email":
            result = agencyzoom_service.search_contact_by_email(arguments.get("email"))
            if result and result.get('contacts'):
                count = len(result['contacts'])
                return f"Found {count} contact(s) in AgencyZoom with email {arguments.get('email')}."
            else:
                return f"No contact found in AgencyZoom with email {arguments.get('email')}."
        
        elif function_name == "submit_collected_data_to_agencyzoom":
            if not insurance_service.insurance_type:
                return "No insurance data has been collected yet. Please collect insurance information first."
            
            insurance_type = insurance_service.insurance_type
            insurance_key = f"{insurance_type}_insurance"
            
            if insurance_key not in insurance_service.collected_data:
                return f"No {insurance_type} insurance data found. Please collect the information first."
            
            insurance_data = insurance_service.collected_data[insurance_key]
            
            # Extract basic info
            full_name = insurance_data.get("full_name", "")
            name_parts = full_name.split(" ", 1)
            first_name = name_parts[0] if name_parts else "Unknown"
            last_name = name_parts[1] if len(name_parts) > 1 else ""
            
            lead_data = {
                "first_name": first_name,
                "last_name": last_name,
                "email": insurance_data.get("email", "noemail@pending.com"),
                "phone": insurance_data.get("phone", ""),
                "insurance_type": insurance_type,
                "source": "AI Chatbot",
                "notes": f"Lead collected via AI chatbot. Thread ID: {thread_id}",
                "insurance_details": insurance_data
            }
            
            result = agencyzoom_service.create_lead(lead_data)
            if result:
                return f"Excellent! I've successfully submitted all your {insurance_type} insurance information to AgencyZoom. Our team will follow up with you shortly!"
            else:
                return "Failed to submit data to AgencyZoom. The information is saved and can be submitted manually."
        
        else:
            return f"Unknown function: {function_name}"
    
    except Exception as e:
        logger.error(f"Error executing function {function_name}: {e}", exc_info=True)
        return f"Error executing {function_name}: {str(e)}"


# ===========================
# CHATBOT ENDPOINTS
# ===========================

@app.post("/chat", response_model=ChatResponse, tags=["Chatbot"])
async def chat(request: ChatRequest):
    """
    Chat endpoint with conversation memory and optional escalation handling.
    
    Args:
        request: ChatRequest containing:
            - query: User's message
            - thread_id: Conversation thread identifier
            - prompt: Optional custom system prompt (if None, uses default CHATBOT_SYSTEM_INSTRUCTIONS)
            - escalation_condition: Optional condition to trigger handover to human
                Example: "user asks to speak with a manager" or "user is frustrated"
            - reset_escalation: Set to True to reset escalation state and continue with bot
                Use this after human interaction is complete
    
    Returns:
        ChatResponse with:
            - response: AI's response message
            - thread_id: Conversation thread identifier
            - timestamp: Response timestamp
            - requires_handover: True if escalation condition is met in this response
            - handover_reason: Explanation why handover is needed
            - escalation_active: True if thread is currently in escalated state
            - escalation_reset: True if escalation was just reset in this request
    """
    try:
        logger.info(f"Received chat request - Thread: {request.thread_id}, Query: {request.query}")
        
        # Check if escalation reset is requested
        escalation_reset = False
        if request.reset_escalation:
            if request.thread_id in thread_escalation_state:
                del thread_escalation_state[request.thread_id]
                logger.info(f"Escalation state reset for thread: {request.thread_id}")
                escalation_reset = True
        
        # Check current escalation state
        current_escalation_state = thread_escalation_state.get(request.thread_id, {})
        escalation_active = current_escalation_state.get("active", False)
        
        # If escalation is active and not reset, inform that handover is required
        if escalation_active and not request.reset_escalation:
            logger.info(f"Thread {request.thread_id} is in escalated state - human handover required")
            return ChatResponse(
                response="This conversation has been escalated to a human agent. Please wait for a human representative to assist you. If you'd like to continue with the AI assistant, please indicate so.",
                thread_id=request.thread_id,
                timestamp=datetime.now().isoformat(),
                requires_handover=True,
                handover_reason=current_escalation_state.get("reason", "Previously escalated"),
                escalation_active=True,
                escalation_reset=False
            )
        
        # Get or create conversation thread with custom prompt if provided
        messages = get_or_create_thread(request.thread_id, custom_prompt=request.prompt)
        
        # Log if custom prompt is being used
        if request.prompt:
            logger.info(f"Using custom prompt for thread: {request.thread_id}")
        
        # Add user message to conversation history
        messages.append({
            "role": "user",
            "content": request.query
        })
        
        # Get available tools
        tools = get_available_tools()
        
        # Call OpenAI API with function calling
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=0.7
        )
        
        assistant_message = response.choices[0].message
        
        # Handle function calls if present
        while assistant_message.tool_calls:
            # Add assistant's response with tool calls to history
            # Note: content can be None when tool calls are present, so we use empty string as fallback
            messages.append({
                "role": "assistant",
                "content": assistant_message.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    } for tc in assistant_message.tool_calls
                ]
            })
            
            # Execute each tool call
            for tool_call in assistant_message.tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)
                
                # Execute the function
                function_result = execute_function_call(
                    function_name,
                    function_args,
                    request.thread_id
                )
                
                # Add function result to messages
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": function_name,
                    "content": function_result
                })
            
            # Get next response from OpenAI
            response = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                tools=tools,
                tool_choice="auto",
                temperature=0.7
            )
            
            assistant_message = response.choices[0].message
        
        # Add final assistant message to history
        messages.append({
            "role": "assistant",
            "content": assistant_message.content or ""
        })
        
        logger.info(f"Chat response generated - Thread: {request.thread_id}")
        
        # Check escalation condition if provided
        requires_handover = False
        handover_reason = None
        
        if request.escalation_condition:
            logger.info(f"Checking escalation condition: {request.escalation_condition}")
            try:
                # Use OpenAI to evaluate if the escalation condition is met
                escalation_check = openai.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {
                            "role": "system",
                            "content": f"""You are an escalation evaluator. Analyze the conversation and determine if the following escalation condition is met:

Escalation Condition: {request.escalation_condition}

Respond with ONLY a JSON object in this exact format:
{{"requires_handover": true/false, "reason": "brief explanation"}}

If the condition is met, set requires_handover to true and provide a reason.
If not met, set requires_handover to false."""
                        },
                        {
                            "role": "user",
                            "content": f"Latest user message: {request.query}\n\nLatest assistant response: {assistant_message.content}"
                        }
                    ],
                    temperature=0.3,
                    response_format={"type": "json_object"}
                )
                
                escalation_result = json.loads(escalation_check.choices[0].message.content)
                requires_handover = escalation_result.get("requires_handover", False)
                handover_reason = escalation_result.get("reason")
                
                if requires_handover:
                    logger.info(f"Escalation triggered - Reason: {handover_reason}")
                    # Store escalation state for this thread
                    thread_escalation_state[request.thread_id] = {
                        "active": True,
                        "reason": handover_reason,
                        "timestamp": datetime.now().isoformat()
                    }
                else:
                    logger.info("Escalation condition not met, continuing conversation")
                    
            except Exception as e:
                logger.error(f"Error checking escalation condition: {e}", exc_info=True)
                # Continue without escalation if check fails
        
        return ChatResponse(
            response=assistant_message.content,
            thread_id=request.thread_id,
            timestamp=datetime.now().isoformat(),
            requires_handover=requires_handover,
            handover_reason=handover_reason,
            escalation_active=thread_escalation_state.get(request.thread_id, {}).get("active", False),
            escalation_reset=escalation_reset
        )
    
    except Exception as e:
        logger.error(f"Error processing chat request: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))



async def update_dynamic_config(
    dynamic_instruction: str = None,
    caller_name: str = None,
    contact_number: str = None,
    language: str = "en",
    voice_id: str = "21m00Tcm4TlvDq8ikWAM",
    transfer_to: str = None,
    escalation_condition: str = None,
    provider: str = "openai",
    api_key: str = None,
):
    """
    Update the dynamic configuration (config.json) with agent parameters.
    
    This replaces the old .env file update approach. The config.json file
    is read by the agent service on each new call/room connection.
    
    Args:
        dynamic_instruction: Custom instructions for the AI agent
        caller_name: Name of the person being called
        contact_number: Phone number being called (for transcript saving)
        language: TTS language (e.g., "en", "es", "fr")
        voice_id: ElevenLabs voice ID (default: Rachel)
        transfer_to: Phone number to transfer to (e.g., +1234567890)
        escalation_condition: Condition when to escalate/transfer the call
        provider: LLM provider ("openai" or "gemini", default: "openai")
        api_key: Custom API key for the provider (optional)
    """
    # Build the full instruction
    if dynamic_instruction:
        if caller_name:
            full_instruction = f"{dynamic_instruction} The caller's name is {caller_name}, address them by name."
        else:
            full_instruction = dynamic_instruction
    else:
        if caller_name:
            full_instruction = f"You are a helpful voice AI assistant. The caller's name is {caller_name}, address them by name."
        else:
            full_instruction = "You are a helpful voice AI assistant."
    
    # Build additional parameters for config
    additional_params = {}
    if contact_number:
        additional_params["contact_number"] = contact_number
    if transfer_to:
        additional_params["transfer_to"] = transfer_to
    if escalation_condition:
        additional_params["escalation_condition"] = escalation_condition
    if provider:
        additional_params["provider"] = provider
    if api_key:
        additional_params["api_key"] = api_key
    
    # Update config.json using the async function
    await update_config_async(
        caller_name=caller_name or "Guest",
        agent_instructions=full_instruction,
        tts_language=language,
        voice_id=voice_id,
        additional_params=additional_params if additional_params else None
    )
    
    log_info(f"Updated config.json with dynamic parameters:")
    log_info(f"  - Agent Instructions: {full_instruction[:100]}...")
    if caller_name:
        log_info(f"  - Caller Name: {caller_name}")
    if contact_number:
        log_info(f"  - Contact Number: {contact_number}")
    log_info(f"  - TTS Language: {language}")
    log_info(f"  - Voice ID: {voice_id}")
    if transfer_to:
        log_info(f"  - Transfer To: {transfer_to}")
    if escalation_condition:
        log_info(f"  - Escalation Condition: {escalation_condition}")
    if provider:
        log_info(f"  - LLM Provider: {provider}")
    if api_key:
        log_info(f"  - Custom API Key: {'***' + api_key[-4:] if len(api_key) > 4 else '***'}")
    


@app.post("/outbound", response_model=StatusResponse, tags=["Outbound Calls"])
async def outbound_call(request: OutboundCallRequest):
    """
    Initiate an outbound call to the specified phone number.
    
    This endpoint:
    1. Validates and initiates the outbound call
    2. Returns immediately with status and caller_id (room name)
    
    Args:
        request: OutboundCallRequest containing:
            - phone_number: Phone number with country code (e.g., +1234567890)
            - name: Caller's name for personalization (optional)
            - dynamic_instruction: Custom instructions for the AI agent (optional)
            - language: TTS language code (default: "en")
            - voice_id: ElevenLabs voice ID (default: Rachel)
            - sip_trunk_id: SIP trunk ID (optional, uses env variable if not provided)
            - transfer_to: Phone number to transfer to (optional)
            - escalation_condition: Condition when to escalate/transfer (optional)
            - provider: LLM provider ("openai" or "gemini", default: "openai")
            - api_key: Custom API key for the provider (optional)
        
    Returns:
        StatusResponse with call status and caller_id (room name)
    
    Example:
        {
            "phone_number": "+1234567890",
            "name": "John Doe",
            "dynamic_instruction": "Ask about appointment",
            "provider": "gemini",
            "api_key": "your-gemini-api-key"
        }
    """
    try:
        log_info(f"Outbound call request to: '{request.phone_number}'")
        
        # Format and validate phone number
        formatted_number = format_phone_number(request.phone_number)
        
        if not validate_phone_number(formatted_number):
            log_error(f"Invalid phone number format: '{request.phone_number}'")
            raise HTTPException(
                status_code=400,
                detail="Invalid phone number format. Phone number must start with '+' followed by country code and number (e.g., +1234567890)"
            )
        
        # Update config.json with dynamic parameters
        log_info("Updating config.json with dynamic parameters...")
        await update_dynamic_config(
            dynamic_instruction=request.dynamic_instruction,
            caller_name=request.name,
            contact_number=formatted_number,  # Pass phone number for transcript saving
            language=request.language,
            voice_id=request.voice_id,
            transfer_to=request.transfer_to,
            escalation_condition=request.escalation_condition,
            provider=request.provider,
            api_key=request.api_key,
        )
        log_info("✓ config.json updated successfully")
        
        log_info(f"Initiating call to formatted number: '{formatted_number}'")
        
        # Make the outbound call and get room name
        participant, room_name = await make_outbound_call(
            phone_number=formatted_number,
            sip_trunk_id=request.sip_trunk_id
        )
        
        log_info(f"Successfully initiated call to '{formatted_number}' for {request.name or 'caller'}")
        log_info(f"Room name (caller_id): {room_name}")
        
        return StatusResponse(
            status="success",
            message=f"Outbound call initiated to {formatted_number}" + (f" for {request.name}" if request.name else ""),
            details={
                "caller_id": room_name,
                "phone_number": formatted_number,
                "original_input": request.phone_number,
                "name": request.name
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        log_exception(f"Error initiating outbound call: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Outbound call error: {str(e)}")


@app.get("/thread/{thread_id}/history", tags=["Chatbot"])
async def get_thread_history(thread_id: str):
    """Get conversation history for a thread."""
    if thread_id not in conversation_threads:
        raise HTTPException(status_code=404, detail=f"Thread {thread_id} not found")
    
    return {
        "thread_id": thread_id,
        "message_count": len(conversation_threads[thread_id]),
        "messages": conversation_threads[thread_id]
    }


@app.delete("/thread/{thread_id}", tags=["Chatbot"])
async def delete_thread(thread_id: str):
    """Delete a conversation thread and its associated services."""
    if thread_id in conversation_threads:
        del conversation_threads[thread_id]
    
    if thread_id in thread_services:
        del thread_services[thread_id]
    
    if thread_id in thread_escalation_state:
        del thread_escalation_state[thread_id]
    
    logger.info(f"Deleted thread: {thread_id}")
    return {"message": f"Thread {thread_id} deleted successfully"}


@app.get("/thread/{thread_id}/escalation", tags=["Chatbot"])
async def get_escalation_status(thread_id: str):
    """
    Get the current escalation status for a conversation thread.
    
    Returns information about whether the conversation is escalated,
    when it was escalated, and the reason for escalation.
    """
    escalation_state = thread_escalation_state.get(thread_id, {})
    
    return {
        "thread_id": thread_id,
        "escalation_active": escalation_state.get("active", False),
        "escalation_reason": escalation_state.get("reason"),
        "escalation_timestamp": escalation_state.get("timestamp"),
        "can_reset": escalation_state.get("active", False)
    }


@app.post("/thread/{thread_id}/escalation/reset", tags=["Chatbot"])
async def reset_escalation_status(thread_id: str):
    """
    Reset the escalation status for a conversation thread.
    
    Use this endpoint to allow the user to continue chatting with the bot
    after human interaction is complete.
    """
    if thread_id in thread_escalation_state:
        del thread_escalation_state[thread_id]
        logger.info(f"Escalation state reset for thread: {thread_id}")
        return {
            "message": "Escalation status reset successfully",
            "thread_id": thread_id,
            "escalation_active": False
        }
    else:
        return {
            "message": "No active escalation found for this thread",
            "thread_id": thread_id,
            "escalation_active": False
        }


# ===========================
# GENERAL ENDPOINTS
# ===========================

@app.get("/health", tags=["General"])
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "active_threads": len(conversation_threads),
        "timestamp": datetime.now().isoformat()
    }


@app.get("/", tags=["General"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Unified Insurance API",
        "version": "1.0.0",
        "description": "Comprehensive API for Insurance Chatbot, SMS, and Email services",
        "services": {
            "chatbot": "AI-powered insurance chatbot with conversation memory",
            "outbound": "AI-powered outbound voice calls via LiveKit SIP",
            "sms": "SMS messaging via Twilio",
            "email": "Email service via SMTP"
        },
        "endpoints": {
            "chatbot": {
                "POST /chat": "Send a message and get a response",
                "GET /thread/{thread_id}/history": "Get conversation history",
                "DELETE /thread/{thread_id}": "Delete a conversation thread"
            },
            "outbound": {
                "POST /outbound": "Initiate an outbound voice call with AI agent"
            },
            "sms": {
                "POST /sms/send": "Send an SMS message",
                "GET /sms/status/{message_sid}": "Get SMS delivery status"
            },
            "email": {
                "POST /email/send": "Send an email"
            },
            "general": {
                "GET /health": "Health check",
                "GET /docs": "Interactive API documentation"
            }
        },
        "documentation": "/docs"
    }


if __name__ == "__main__":
    import uvicorn
    
    logger.info("Starting Unified Insurance API server...")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

