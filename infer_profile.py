import aiosqlite
import aiohttp
import asyncio
import logging
import time
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from telethon import TelegramClient, events, utils, errors
from telethon.tl.types import PeerUser
from telethon.tl.functions.users import GetFullUserRequest

# Setup logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO,
)

# Try to import configuration
try:
    from config import *
except ImportError:
    logging.error("Configuration file 'config.py' not found!")
    logging.error("Please copy 'config.example.py' to 'config.py' and fill in your credentials.")
    exit(1)

# Initialize random seed
torch.manual_seed(SEED)

MISSING = '__MISSING__'

# Model initialization
# Use MPS if possible on macbook with M chip
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-multilingual-cased', num_labels=2)
model = model.to(device)
model.load_state_dict(torch.load(USER_MODEL_PATH, map_location=device))
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-multilingual-cased')

async def init_db():
    """
    Initializes the SQLite database and creates the user_data table if it doesn't exist.
    """
    async with aiosqlite.connect(USER_DB_PATH) as conn:
        await conn.execute("""
        CREATE TABLE IF NOT EXISTS user_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tg_id INTEGER,
            nickname TEXT,
            username TEXT,
            bio TEXT,
            is_premium BOOLEAN,
            model_score REAL,
            is_spam BOOLEAN,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)

async def insert_user_data(tg_id, nickname, username, bio, is_premium, model_score, is_spam=False):
    """
    Inserts a new user's data into the SQLite database.
    """
    async with aiosqlite.connect(USER_DB_PATH) as db:
        await db.execute("""
            INSERT INTO user_data (tg_id, nickname, username, bio, is_premium, model_score, is_spam)
            VALUES (?, ?, ?, ?, ?, ?, ?);
        """, (tg_id, nickname, username, bio, is_premium, model_score, is_spam))
        await db.commit()

async def is_user_processed(user_id):
    """
    Checks if a user with the given Telegram user ID (tg_id) already exists in the database.
    """
    query = "SELECT 1 FROM user_data WHERE tg_id = ? LIMIT 1;"
    async with aiosqlite.connect(USER_DB_PATH) as db:
        async with db.execute(query, (user_id,)) as cursor:
            result = await cursor.fetchone()
            return result is not None

client = TelegramClient('spam_detector', API_ID, API_HASH)

async def api(session: aiohttp.ClientSession, action: str, data: dict, timeout: float = 60):
    try:
        async with session.post(API_URL + action, json=data, timeout=timeout + 10) as response:
            resp = await response.json()
            if not resp['ok']:
                raise RuntimeError(f'Telegram Bot API request "{action}" failed: {resp["description"]}')
            return resp['result']
    except aiohttp.ClientError as e:
        raise RuntimeError(f"HTTP request error: {e}")
    except asyncio.TimeoutError:
        return []

async def get_updates(offset: int = 0, timeout: float = 60):
    async with aiohttp.ClientSession() as session:
        while True:
            try:
                update_data = {
                    'offset': offset,
                    'timeout': timeout,
                    'allowed_updates': ['chat_member'],
                }
                updates = await api(session, 'getUpdates', update_data, timeout=timeout)
                for update in updates:
                    offset = update['update_id'] + 1  # Update offset to avoid re-fetching
                    yield update
            except RuntimeError as e:
                logging.warning(f"Error fetching updates: {e.args[0]}")

async def get_model_score(nickname, username, bio, is_premium=False):
    # TODO: handle is_premium
    def concat_features(nickname, username, bio, is_premium=False):
        return f'{nickname} [SEP] {username} [SEP] {bio}'

    # Tokenize usernames
    encoded_usernames = tokenizer([
        concat_features(nickname, username, bio, is_premium)
    ], padding=True, truncation=True, max_length=512, return_tensors='pt')
    input_ids = encoded_usernames['input_ids'].to(device)
    attention_mask = encoded_usernames['attention_mask'].to(device)

    # Prediction
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probs = torch.sigmoid(logits)[:, 1]
    return probs.cpu().numpy()[0]

def is_spam(model_score):
    return (model_score >= USER_SPAM_THRESHOLD)

async def handle_update(update):
    if 'chat_member' in update:
        chat_member = update['chat_member']
        chat = chat_member['chat']
        # Chat must be whitelisted
        if chat['id'] not in USER_GROUP_LIST and chat.get('username', '').lower() not in USER_GROUP_LIST:
            return
        new_chat_member = chat_member['new_chat_member']
        old_chat_member = chat_member['old_chat_member']
        user = new_chat_member['user']
        user_id = user['id']
        # Ignore all bots
        if user['is_bot']:
            return
        # User must be actually joining, not being restricted/kicked/etc.
        if new_chat_member['status'] == 'member' or new_chat_member.get('is_member'):
            if (old_chat_member['status'] == 'member' or old_chat_member.get('is_member')):
                return
        else:
            return
        # Already seen this user?
        if await is_user_processed(user_id):
            logging.info(f"User {user_id} already processed, skipping.")
            return

        is_premium = user.get('is_premium', False)
        logging.info(update)
        try:
            # To "encounter" the newly joined ID
            await client.get_participants(chat['id'], limit=20)
            entity = await client.get_entity(PeerUser(user_id))
            full_user = await client(GetFullUserRequest(PeerUser(user_id)))
        except (ValueError, errors.FloodWaitError) as e:
            logging.warning(f'Unable to get entity of {user_id}: {e}')
            username = user.get('username', MISSING)
            nickname = user.get('first_name', '') + (f" {user.get('last_name', '')}" if user.get('last_name') else '')
            nickname = nickname or MISSING
            bio = MISSING
        else:
            username = entity.username or MISSING
            nickname = utils.get_display_name(entity) or MISSING
            bio = full_user.full_user.about or MISSING
        logging.info(f"[{update['update_id']}] New user joined in {chat.get('username', chat['id'])}: username={username}, nickname={nickname}, bio={bio}, is_premium={is_premium}. Update is lagging by {time.time() - int(chat_member.get('date'))} seconds")

        model_score = await get_model_score(nickname, username, bio, is_premium=is_premium)
        if is_spam(model_score):
            logging.info(f"⚠️ Detected spammer: {nickname} (probability {model_score:.4f})")
        else:
            logging.info(f"User {nickname} is not a spammer (probability {model_score:.4f}).")
        await insert_user_data(user_id, nickname, username, bio, is_premium, str(model_score), is_spam(model_score))

        if not is_spam(model_score):
            return

        # Run global ban and message deletion
        try:
            messages = await client.get_messages(chat.get('username'), 10, from_user=PeerUser(user_id))
            await client.delete_messages(chat.get('username'), messages)
            if USER_CAPTCHA_BOT_ID:
                messages = await client.get_messages(chat.get('username'), 10, from_user=PeerUser(USER_CAPTCHA_BOT_ID), search=str(user_id))
                await client.delete_messages(chat.get('username'), messages)
        except Exception as e:
            logging.warning(f'Unable to delete messages of {user_id}: {e}')
        try:
            await client.send_message(USER_BAN_CHANNEL, USER_BAN_COMMAND.format(user_id=user_id, model_score=model_score))
        except Exception as e:
            logging.warning(f'Unable to send gbb message of {user_id}: {e}')

async def listen_for_updates():
    async for update in get_updates():
        asyncio.create_task(handle_update(update))

@client.on(events.NewMessage(chats=[USER_BAN_CHANNEL], pattern=r"^/wl (\d+)"))
async def handle_wl_command(event):
    """
    Listens for /wl user_id commands in the USER_BAN_CHANNEL group.
    Sets the is_spam field to False for the given user_id in the database.
    """
    user_id = int(event.pattern_match.group(1))  # Extract user_id from command

    async with aiosqlite.connect(USER_DB_PATH) as db:
        # Check if the user exists in the database
        async with db.execute("SELECT is_spam FROM user_data WHERE tg_id = ?", (user_id,)) as cursor:
            row = await cursor.fetchone()

        if row is None:
            # User does not exist
            await event.reply(f"❌ User with ID {user_id} not found in the database.")
            return

        current_result = row[0]  # is_spam column

        if current_result is False:
            # Already marked as not spam
            await event.reply(f"⚠️ User with ID {user_id} is already marked as not spam.")
            return

        # Update the is_spam field to False
        await db.execute(
            "UPDATE user_data SET is_spam = ? WHERE tg_id = ?;",
            (False, user_id)
        )
        await db.commit()

    # Reply with success message
    await event.reply(f"✅ User with ID {user_id} successfully marked as not spam.")
    await event.reply(USER_UNBAN_COMMAND.format(user_id=user_id))


async def main():
    await init_db()
    await asyncio.gather(
        listen_for_updates(),
        client.run_until_disconnected(),
        # Add other coroutines here if needed
    )


if __name__ == '__main__':
    try:
        with client:
            client.loop.run_until_complete(main())
    except KeyboardInterrupt:
        exit(0)
