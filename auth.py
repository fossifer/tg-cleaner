from fastapi import HTTPException, Request
from starlette.responses import RedirectResponse
import hashlib
import hmac
import time
import json
from typing import Optional, Dict, Any
from admin_db import get_user, add_user
import ipaddress

def is_local_request(request: Request) -> bool:
    """Check if the request is from localhost or 127.0.0.1."""
    client_host = getattr(request.client, 'host', None)
    if not client_host:
        return False
    try:
        ip = ipaddress.ip_address(client_host)
        return ip.is_loopback
    except ValueError:
        # If not an IP, check for 'localhost' string
        return client_host == 'localhost'

class TelegramAuth:
    def __init__(self, bot_token: str):
        self.bot_token = bot_token
        self.bot_token_hash = hashlib.sha256(bot_token.encode()).hexdigest()

    def verify_telegram_data(self, auth_data: Dict[str, Any]) -> bool:
        """Verify the authentication data received from Telegram."""
        if not auth_data:
            return False

        # Check if the data is not older than 24 hours
        auth_date = int(auth_data.get('auth_date', 0))
        if time.time() - auth_date > 86400:
            return False

        # Create data check string
        check_string = '\n'.join(f'{k}={v}' for k, v in sorted(
            auth_data.items(),
            key=lambda x: x[0]
        ) if k != 'hash')

        # Calculate hash
        secret_key = hashlib.sha256(self.bot_token.encode()).digest()
        hash = hmac.new(
            secret_key,
            check_string.encode(),
            hashlib.sha256
        ).hexdigest()

        return hash == auth_data.get('hash')

    async def get_current_user(self, request: Request) -> Optional[Dict[str, Any]]:
        # Bypass authentication for local development (127.0.0.1 / localhost)
        if is_local_request(request):
            # Return a fake super_admin user when running locally
            return {
                "telegram_id": 0,
                "username": "dev_local",
                "role": "super_admin"
            }
        auth_data = request.session.get('auth_data')
        if not auth_data:
            return None

        # Verify the auth data
        if not self.verify_telegram_data(auth_data):
            request.session.pop('auth_data', None)
            return None

        # Get or create user
        user = await get_user(auth_data['id'])
        if not user:
            # Auto-register new users
            user = await add_user(
                telegram_id=auth_data['id'],
                username=auth_data.get('username'),
                role='user'  # Default role
            )

        return user

    async def require_role(self, request: Request, required_role: str) -> Dict[str, Any]:
        # Bypass authentication for local development (127.0.0.1 / localhost)
        if is_local_request(request):
            # Return a fake super_admin user when running locally
            return {
                "telegram_id": 0,
                "username": "dev_local",
                "role": "super_admin"
            }
        user = await self.get_current_user(request)
        if not user:
            # Store the original URL to redirect back after login
            request.session['redirect_url'] = str(request.url)
            raise HTTPException(
                status_code=302,
                detail="Login required",
                headers={"Location": "/admin/login"}
            )

        if user['role'] != required_role and user['role'] != 'super_admin':
            raise HTTPException(status_code=403, detail="Insufficient permissions")

        return user

    async def require_admin(self, request: Request) -> Dict[str, Any]:
        # Bypass authentication for local development (127.0.0.1 / localhost)
        if is_local_request(request):
            # Return a fake super_admin user when running locally
            return {
                "telegram_id": 0,
                "username": "dev_local",
                "role": "super_admin"
            }
        user = await self.get_current_user(request)
        if not user:
            # Store the original URL to redirect back after login
            request.session['redirect_url'] = str(request.url)
            raise HTTPException(
                status_code=302,
                detail="Login required",
                headers={"Location": "/admin/login"}
            )

        if user['role'] not in ['admin', 'super_admin']:
            raise HTTPException(status_code=403, detail="Admin access required")

        return user

    async def require_super_admin(self, request: Request) -> Dict[str, Any]:
        # Bypass authentication for local development (127.0.0.1 / localhost)
        if is_local_request(request):
            # Return a fake super_admin user when running locally
            return {
                "telegram_id": 0,
                "username": "dev_local",
                "role": "super_admin"
            }
        user = await self.get_current_user(request)
        if not user:
            # Store the original URL to redirect back after login
            request.session['redirect_url'] = str(request.url)
            raise HTTPException(
                status_code=302,
                detail="Login required",
                headers={"Location": "/admin/login"}
            )

        if user['role'] != 'super_admin':
            raise HTTPException(status_code=403, detail="Super admin access required")

        return user 