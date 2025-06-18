import aiosqlite
from datetime import datetime
from typing import Optional, List, Dict, Any
import json
import os

# Get database directory from environment variable, default to current directory
DB_DIR = os.getenv('DB_DIR', os.getcwd())
ADMIN_DB_PATH = os.path.join(DB_DIR, "admin_data.db")

async def init_db(db_path: str = ADMIN_DB_PATH):
    async with aiosqlite.connect(db_path) as db:
        # Users table
        await db.execute("""
            CREATE TABLE IF NOT EXISTS users (
                telegram_id INTEGER PRIMARY KEY,
                username TEXT,
                role TEXT CHECK(role IN ('super_admin', 'admin', 'user')),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP
            )
        """)
        
        # Models table
        await db.execute("""
            CREATE TABLE IF NOT EXISTS models (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                type TEXT CHECK(type IN ('transformer', 'traditional', 'nsfw')),
                file_path TEXT NOT NULL,
                upload_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                is_active BOOLEAN DEFAULT 0,
                metadata TEXT
            )
        """)
        
        # Model switch history
        await db.execute("""
            CREATE TABLE IF NOT EXISTS model_switches (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id INTEGER,
                switched_by INTEGER,
                switched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (model_id) REFERENCES models(id),
                FOREIGN KEY (switched_by) REFERENCES users(telegram_id)
            )
        """)
        
        # System status logs
        await db.execute("""
            CREATE TABLE IF NOT EXISTS system_status (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                component TEXT NOT NULL,
                status TEXT NOT NULL,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                details TEXT
            )
        """)
        
        await db.commit()

async def add_user(telegram_id: int, username: str, role: str = 'user', db_path: str = ADMIN_DB_PATH) -> bool:
    async with aiosqlite.connect(db_path) as db:
        try:
            await db.execute(
                "INSERT INTO users (telegram_id, username, role) VALUES (?, ?, ?)",
                (telegram_id, username, role)
            )
            await db.commit()
            return True
        except aiosqlite.IntegrityError:
            return False

async def get_user(telegram_id: int, db_path: str = ADMIN_DB_PATH) -> Optional[Dict[str, Any]]:
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT * FROM users WHERE telegram_id = ?",
            (telegram_id,)
        )
        row = await cursor.fetchone()
        return dict(row) if row else None

async def update_user_role(telegram_id: int, new_role: str, db_path: str = ADMIN_DB_PATH) -> bool:
    async with aiosqlite.connect(db_path) as db:
        try:
            await db.execute(
                "UPDATE users SET role = ? WHERE telegram_id = ?",
                (new_role, telegram_id)
            )
            await db.commit()
            return True
        except aiosqlite.Error:
            return False

async def add_model(name: str, model_type: str, file_path: str, metadata: Optional[Dict] = None, db_path: str = ADMIN_DB_PATH) -> int:
    async with aiosqlite.connect(db_path) as db:
        cursor = await db.execute(
            "INSERT INTO models (name, type, file_path, metadata) VALUES (?, ?, ?, ?)",
            (name, model_type, file_path, json.dumps(metadata) if metadata else None)
        )
        await db.commit()
        return cursor.lastrowid

async def get_active_model(db_path: str = ADMIN_DB_PATH) -> Optional[Dict[str, Any]]:
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT * FROM models WHERE is_active = 1"
        )
        row = await cursor.fetchone()
        return dict(row) if row else None

async def switch_active_model(model_id: int, switched_by: int, db_path: str = ADMIN_DB_PATH) -> bool:
    async with aiosqlite.connect(db_path) as db:
        try:
            await db.execute("BEGIN TRANSACTION")
            
            # Deactivate all models
            await db.execute("UPDATE models SET is_active = 0")
            
            # Activate the new model
            await db.execute(
                "UPDATE models SET is_active = 1 WHERE id = ?",
                (model_id,)
            )
            
            # Record the switch
            await db.execute(
                "INSERT INTO model_switches (model_id, switched_by) VALUES (?, ?)",
                (model_id, switched_by)
            )
            
            await db.commit()
            return True
        except aiosqlite.Error:
            await db.rollback()
            return False

async def get_model_switch_history(limit: int = 10, db_path: str = ADMIN_DB_PATH) -> List[Dict[str, Any]]:
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute("""
            SELECT ms.*, m.name as model_name, u.username as switched_by_username
            FROM model_switches ms
            JOIN models m ON ms.model_id = m.id
            JOIN users u ON ms.switched_by = u.telegram_id
            ORDER BY ms.switched_at DESC
            LIMIT ?
        """, (limit,))
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]

async def update_system_status(component: str, status: str, details: Optional[Dict] = None, db_path: str = ADMIN_DB_PATH):
    async with aiosqlite.connect(db_path) as db:
        await db.execute("""
            INSERT INTO system_status (component, status, details)
            VALUES (?, ?, ?)
        """, (component, status, json.dumps(details) if details else None))
        await db.commit()

async def get_system_status(db_path: str = ADMIN_DB_PATH) -> Dict[str, Any]:
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute("""
            SELECT component, status, details, last_updated
            FROM system_status
            ORDER BY last_updated DESC
        """)
        rows = await cursor.fetchall()
        return {row['component']: {
            'status': row['status'],
            'details': json.loads(row['details']) if row['details'] else None,
            'last_updated': row['last_updated']
        } for row in rows} 