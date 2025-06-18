from fastapi import FastAPI, Request, Form, File, UploadFile, Depends, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.sessions import SessionMiddleware
import aiosqlite
import os
import json
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
import asyncio
import subprocess
from pathlib import Path
# from config import SECRET_KEY, WEB_BOT_USERNAME
SECRET_KEY = os.getenv('SECRET_KEY')
WEB_BOT_USERNAME = os.getenv('WEB_BOT_USERNAME')
from auth import TelegramAuth
from admin_db import (
    init_db, add_model, get_active_model, switch_active_model,
    get_model_switch_history, update_system_status, get_system_status,
    get_user, update_user_role, add_user
)
from starlette.middleware.authentication import AuthenticationMiddleware
from starlette.authentication import AuthenticationBackend

# Get database directory from environment variable, default to current directory
DB_DIR = os.getenv('DB_DIR', os.getcwd())

# Database file paths
MESSAGE_DB_PATH = os.path.join(DB_DIR, "message_data.db")
USER_DB_PATH = os.path.join(DB_DIR, "user_data.db")
ADMIN_DB_PATH = os.path.join(DB_DIR, "admin_data.db")

def get_utc_now() -> str:
    """Get current UTC time in ISO format with timezone."""
    return datetime.now(timezone.utc).isoformat()

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["Content-Security-Policy"] = "frame-ancestors 'self' http://127.0.0.1:* http://localhost:* https://telegram.org;"
        return response

app = FastAPI()
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(SessionMiddleware, secret_key=SECRET_KEY)
# # More here https://www.starlette.io/authentication/
# app.add_middleware(AuthenticationMiddleware, backend=AuthenticationBackend)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Initialize auth with bot token from config
from config import WEB_BOT_TOKEN
auth = TelegramAuth(WEB_BOT_TOKEN)

# Initialize database on startup
@app.on_event("startup")
async def startup_event():
    await init_db(ADMIN_DB_PATH)

# Admin panel routes
@app.get("/admin", response_class=HTMLResponse)
async def admin_panel(request: Request):
    user = await auth.get_current_user(request)
    if not user:
        request.session['redirect_url'] = str(request.url)
        raise HTTPException(
            status_code=302,
            detail="Login required",
            headers={"Location": "/admin/login"}
        )
    
    # Get database stats
    async with aiosqlite.connect(MESSAGE_DB_PATH) as db:
        cursor = await db.execute("SELECT COUNT(*) FROM message_data")
        total_messages = (await cursor.fetchone())[0]
        
        cursor = await db.execute("SELECT COUNT(*) FROM message_data WHERE is_spam IN (1, X'01')")
        spam_messages = (await cursor.fetchone())[0]
    
    # Get user stats
    async with aiosqlite.connect(USER_DB_PATH) as db:
        cursor = await db.execute("SELECT COUNT(*) FROM user_data")
        total_users = (await cursor.fetchone())[0]
        
        cursor = await db.execute("SELECT COUNT(*) FROM user_data WHERE is_spam IN (1, X'01')")
        spam_users = (await cursor.fetchone())[0]
    
    # Get active model
    active_model = await get_active_model(ADMIN_DB_PATH)
    
    return templates.TemplateResponse("admin/index.html", {
        "request": request,
        "user": user,
        "total_messages": total_messages,
        "spam_messages": spam_messages,
        "total_users": total_users,
        "spam_users": spam_users,
        "active_model": active_model
    })

@app.get("/admin/status", response_class=HTMLResponse)
async def system_status(request: Request):
    user = await auth.get_current_user(request)
    status = await get_system_status(ADMIN_DB_PATH)
    
    # Get database stats from the status
    db_status = status.get('database', {})
    details = db_status.get('details', {})

    # Log paths
    msg_log_path = os.getenv('MSG_LOG_PATH')
    user_log_path = os.getenv('USER_LOG_PATH')
    infer_msgs_log = None
    infer_profile_log = None
    log_error = None
    can_view_logs = user and user.get('role') in ['admin', 'super_admin']
    if can_view_logs:
        def tail_log(path):
            if not path or not os.path.exists(path):
                return None
            try:
                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = f.readlines()
                    return ''.join(lines[-500:])
            except Exception as e:
                return f"[Error reading log: {e}]"
        infer_msgs_log = tail_log(msg_log_path)
        infer_profile_log = tail_log(user_log_path)
    else:
        log_error = "Only admins can view log content."

    # Get recent model switch history
    model_switches = await get_model_switch_history(10, ADMIN_DB_PATH)

    return templates.TemplateResponse("admin/status.html", {
        "request": request,
        "user": user,
        "status": status,
        "total_messages": details.get('total_messages', 0),
        "spam_messages": details.get('spam_messages', 0),
        "total_users": details.get('total_users', 0),
        "spam_users": details.get('spam_users', 0),
        "infer_msgs_log": infer_msgs_log,
        "infer_profile_log": infer_profile_log,
        "log_error": log_error,
        "model_switches": model_switches
    })

@app.get("/admin/models", response_class=HTMLResponse)
async def model_management(request: Request):
    user = await auth.require_admin(request)
    
    async with aiosqlite.connect(ADMIN_DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute("SELECT * FROM models ORDER BY upload_time DESC")
        models = [dict(row) for row in await cursor.fetchall()]
        
        active_model = await get_active_model(ADMIN_DB_PATH)
    
    return templates.TemplateResponse("admin/models.html", {
        "request": request,
        "user": user,
        "models": models,
        "active_model": active_model
    })

@app.post("/admin/models/upload")
async def upload_model(
    request: Request,
    file: UploadFile = File(...),
    model_name: str = Form(...),
    model_type: str = Form(...),
    metadata: Optional[str] = Form(None)
):
    user = await auth.require_admin(request)
    
    # Create models directory if it doesn't exist
    models_dir = os.path.join(DB_DIR, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    # Save the file
    file_path = os.path.join(models_dir, file.filename)
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)
    
    # Parse metadata if provided
    metadata_dict = json.loads(metadata) if metadata else None
    
    # Add model to database
    model_id = await add_model(
        name=model_name,
        model_type=model_type,
        file_path=file_path,
        metadata=metadata_dict,
        db_path=ADMIN_DB_PATH
    )
    
    return {"success": True, "model_id": model_id}

@app.post("/admin/models/switch")
async def switch_model(request: Request, model_id: int = Form(...)):
    user = await auth.require_admin(request)
    
    success = await switch_active_model(model_id, user['telegram_id'], ADMIN_DB_PATH)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to switch model")
    
    return {"success": True}

@app.get("/admin/test", response_class=HTMLResponse)
async def model_test(request: Request):
    user = await auth.require_admin(request)
    
    async with aiosqlite.connect(ADMIN_DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute("SELECT * FROM models WHERE is_active = 1")
        active_models = [dict(row) for row in await cursor.fetchall()]
    
    return templates.TemplateResponse("admin/test.html", {
        "request": request,
        "user": user,
        "active_models": active_models
    })

@app.post("/admin/test/predict")
async def test_prediction(
    request: Request,
    nickname: Optional[str] = Form(None),
    message_text: Optional[str] = Form(None)
):
    user = await auth.require_admin(request)
    
    # Get active models
    active_models = await get_active_model(ADMIN_DB_PATH)
    if not active_models:
        raise HTTPException(status_code=400, detail="No active models found")
    
    # Run predictions for each model
    results = []
    for model in active_models:
        # Here you would implement the actual prediction logic
        # This is a placeholder that should be replaced with actual model inference
        result = {
            "model_name": model["name"],
            "model_type": model["type"],
            "prediction": {
                "is_spam": False,  # Placeholder
                "confidence": 0.5,  # Placeholder
                "details": {}  # Placeholder for additional model-specific details
            }
        }
        results.append(result)
    
    return {"results": results}

@app.get("/admin/users", response_class=HTMLResponse)
async def user_management(request: Request):
    user = await auth.require_super_admin(request)
    
    async with aiosqlite.connect(ADMIN_DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute("SELECT * FROM users ORDER BY created_at DESC")
        users = [dict(row) for row in await cursor.fetchall()]
    
    return templates.TemplateResponse("admin/users.html", {
        "request": request,
        "user": user,
        "users": users
    })

@app.post("/admin/users/update-role")
async def update_user_role_endpoint(
    request: Request,
    telegram_id: int = Form(...),
    new_role: str = Form(...)
):
    user = await auth.require_super_admin(request)
    
    success = await update_user_role(telegram_id, new_role, ADMIN_DB_PATH)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to update user role")
    
    return {"success": True}

# Login routes
@app.get("/admin/login", response_class=HTMLResponse)
async def login_page(request: Request):
    # If user is already logged in, redirect to admin panel
    user = await auth.get_current_user(request)
    if user:
        return RedirectResponse(url="/admin")
    
    return templates.TemplateResponse("admin/login.html", {
        "request": request,
        "bot_username": WEB_BOT_USERNAME,
        "user": {
            "telegram_id": 0,
            "username": "[visitor]",
            "role": "user"
        }
    })

@app.post("/admin/login")
async def login(request: Request):
    auth_data = await request.json()
    
    # Verify the auth data
    if not auth.verify_telegram_data(auth_data):
        return JSONResponse({
            "success": False,
            "error": "Invalid authentication data"
        })
    
    # Store auth data in session
    request.session['auth_data'] = auth_data
    
    # Get or create user
    user = await get_user(auth_data['id'])
    if not user:
        user = await add_user(
            telegram_id=auth_data['id'],
            username=auth_data.get('username'),
            role='user'  # Default role
        )
    
    # Get redirect URL from session or default to admin panel
    redirect_url = request.session.pop('redirect_url', '/admin')
    
    return JSONResponse({
        "success": True,
        "redirect_url": redirect_url
    })

@app.get("/admin/logout")
async def logout(request: Request):
    request.session.pop('auth_data', None)
    return RedirectResponse(url="/admin/login")

# Background task to update system status
async def update_system_status_task():
    while True:
        try:
            # Update database stats
            async with aiosqlite.connect(MESSAGE_DB_PATH) as db:
                cursor = await db.execute("SELECT COUNT(*) FROM message_data")
                total_messages = (await cursor.fetchone())[0]
                
                cursor = await db.execute("SELECT COUNT(*) FROM message_data WHERE is_spam IN (1, X'01')")
                spam_messages = (await cursor.fetchone())[0]

                # Get latest message time
                cursor = await db.execute("SELECT created_at FROM message_data ORDER BY created_at DESC LIMIT 1")
                last_message = await cursor.fetchone()
                last_message_time = last_message[0] if last_message else None

            # Get user stats
            async with aiosqlite.connect(USER_DB_PATH) as db:
                cursor = await db.execute("SELECT COUNT(*) FROM user_data")
                total_users = (await cursor.fetchone())[0]
                
                cursor = await db.execute("SELECT COUNT(*) FROM user_data WHERE is_spam IN (1, X'01')")
                spam_users = (await cursor.fetchone())[0]

                # Get latest user time
                cursor = await db.execute("SELECT created_at FROM user_data ORDER BY created_at DESC LIMIT 1")
                last_user = await cursor.fetchone()
                last_user_time = last_user[0] if last_user else None
            
            await update_system_status(
                "database",
                "ok",
                {
                    "total_messages": total_messages,
                    "spam_messages": spam_messages,
                    "total_users": total_users,
                    "spam_users": spam_users,
                    "last_message_time": last_message_time,
                    "last_user_time": last_user_time,
                    "last_updated": get_utc_now()
                },
                ADMIN_DB_PATH
            )
            
        except Exception as e:
            print(f"Error updating system status: {e}")
        
        await asyncio.sleep(60)  # Update every minute

# Start the background task
@app.on_event("startup")
async def start_background_tasks():
    asyncio.create_task(update_system_status_task()) 