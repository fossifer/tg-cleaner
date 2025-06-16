from fastapi import FastAPI, Request, Form, Depends
from fastapi.responses import HTMLResponse, StreamingResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import aiosqlite
import csv
import io
from datetime import datetime
from typing import Optional

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

DATABASE_PATH = "message_data.db"
PAGE_SIZE = 20

@app.get("/", response_class=HTMLResponse)
async def read_data(request: Request, page: int = 1):
    offset = (page - 1) * PAGE_SIZE
    async with aiosqlite.connect(DATABASE_PATH) as db:
        cursor = await db.execute(
            "SELECT COUNT(*) FROM message_data"
        )
        total_rows = (await cursor.fetchone())[0]

        cursor = await db.execute(
            "SELECT id, user_id, nickname, message_text, model_score, is_spam, created_at FROM message_data ORDER BY created_at DESC LIMIT ? OFFSET ?",
            (PAGE_SIZE, offset)
        )
        rows = await cursor.fetchall()

    processed_rows = []
    for row in rows:
        is_spam = row[5]
        if isinstance(is_spam, bytes):
            is_spam = is_spam == b'\x01'
        processed_rows.append({
            "id": row[0],
            "user_id": row[1],
            "nickname": row[2],
            "message_text": row[3],
            "model_score": row[4],
            "is_spam": is_spam,
            "created_at": row[6],
        })

    total_pages = (total_rows + PAGE_SIZE - 1) // PAGE_SIZE

    return templates.TemplateResponse("index.html", {
        "request": request,
        "rows": processed_rows,
        "current_page": page,
        "total_pages": total_pages,
    })

@app.post("/update_label")
async def update_label(id: int = Form(...), is_spam: bool = Form(...)):
    async with aiosqlite.connect(DATABASE_PATH) as db:
        await db.execute("UPDATE message_data SET is_spam = ? WHERE id = ?", (is_spam, id))
        await db.commit()
    return {"success": True}

@app.get("/export", response_class=StreamingResponse)
async def export_csv(after: Optional[str] = None):
    async with aiosqlite.connect(DATABASE_PATH) as db:
        if after:
            query = "SELECT user_id, nickname, message_text, is_spam FROM message_data WHERE created_at >= ?"
            cursor = await db.execute(query, (after,))
        else:
            cursor = await db.execute("SELECT user_id, nickname, message_text, is_spam FROM message_data")
        rows = await cursor.fetchall()

    def generate():
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["user_id", "nickname", "message_text", "is_spam"])
        for row in rows:
            is_spam = row[3]
            if isinstance(is_spam, bytes):
                is_spam = is_spam == b'\x01'
            writer.writerow([row[0], row[1], row[2], is_spam])
        output.seek(0)
        return output.read()

    return StreamingResponse(io.StringIO(generate()), media_type="text/csv", headers={
        "Content-Disposition": "attachment; filename=export.csv"
    })

@app.get("/messages_fragment", response_class=HTMLResponse)
async def messages_fragment(
    request: Request,
    search: str = '',
    page: int = 1,
):
  async with aiosqlite.connect(DATABASE_PATH) as db:
    offset = (page - 1) * PAGE_SIZE
    like_params = [f"%{search}%"] * 3
    total_rows = 0

    if search:
        query = f"""
        SELECT id, user_id, nickname, message_text, model_score, is_spam, created_at FROM message_data
        WHERE 
            CAST(user_id AS TEXT) LIKE ? OR 
            nickname LIKE ? OR 
            message_text LIKE ?
        ORDER BY created_at DESC LIMIT ? OFFSET ?
        """
        values = like_params + [PAGE_SIZE + 1, offset]
        cursor = await db.execute(query, values)
    else:
        cursor = await db.execute(
            "SELECT id, user_id, nickname, message_text, model_score, is_spam, created_at FROM message_data ORDER BY created_at DESC LIMIT ? OFFSET ?",
            (PAGE_SIZE + 1, offset)
        )

    rows = await cursor.fetchall()
    has_next = len(rows) > PAGE_SIZE
    total_pages = (len(rows) + PAGE_SIZE - 1) // PAGE_SIZE
    rows = rows[:PAGE_SIZE]

    def parse_bool(val):
        return val in (1, True, b'\x01')

    messages = [
        {
            "id": row[0],
            "user_id": row[1],
            "nickname": row[2],
            "message_text": row[3],
            "model_score": row[4],
            "is_spam": parse_bool(row[5]),
            "created_at": row[6],
        }
        for row in rows
    ]

    return templates.TemplateResponse("messages_table.html", {
        "request": request,
        "rows": messages,
        "page": page,
        "has_next": has_next,
        "current_page": page,
        "total_pages": total_pages,
    })
