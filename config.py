"""Configuration settings for price monitoring system"""

API_KEYS = {
    'data_gov': '35985678-0d79-46b4-9ed6-6f13308a1d24',  # Get from https://data.gov.in/
    'agmarknet': 'YOUR_AGMARKNET_API_KEY',  # Get from Agmarknet portal
}

# Additional API configurations
API_CONFIG = {
    'retry_attempts': 3,
    'timeout': 30,
    'cache_duration': 3600  # Cache API responses for 1 hour
}