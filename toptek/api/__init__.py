"""FastAPI wiring for ProjectX gateway integration."""

from .models import GatewaySettings, load_gateway_settings
from .routes_gateway import register_gateway_routes

__all__ = ["GatewaySettings", "load_gateway_settings", "register_gateway_routes"]
