import aiosqlite
import asyncio
import datetime
import io
import logging
import os
import time
import torch
from PIL import Image
from transformers import AutoModelForSequenceClassification, AutoTokenizer, ViTForImageClassification, ViTFeatureExtractor
from telethon import TelegramClient, events, utils
from telethon.tl.types import PeerUser
from train_msgs_nb import TelegramMessageClassifier

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

# Model initialization
# Use MPS if possible on macbook with M chip
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-multilingual-cased', num_labels=2)
model = model.to(device)
model.load_state_dict(torch.load(MSG_MODEL_PATH))
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-multilingual-cased')

# Naive Bayes
classifier = TelegramMessageClassifier()
classifier.load('models/nb_model_and_vectorizer.joblib')

# NSFW Image Detection
feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
nsfw_model = ViTForImageClassification.from_pretrained(NSFW_MODEL_PATH).to(device)
nsfw_model.eval()

async def classify_image(image_data, model, feature_extractor):
    try:
        image = Image.open(io.BytesIO(image_data))
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        inputs = feature_extractor(images=image, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        positive_probability = probabilities[0, 1].item()
        
        return {
            "is_nsfw": positive_probability > MSG_NSFW_THRESHOLD,
            "nsfw_probability": positive_probability
        }
    except Exception as e:
        logging.error(f"Error classifying image: {str(e)}")
        return {"is_nsfw": False, "nsfw_probability": 0.0, "error": str(e)}

# Initialize client
client = TelegramClient('message_spam_detector', API_ID, API_HASH)

# Database initialization
async def init_db():
    """
    Initialize SQLite database with tables for message and user activity tracking
    """
    async with aiosqlite.connect(MSG_DB_PATH) as conn:
        # Table for message data
        await conn.execute("""
        CREATE TABLE IF NOT EXISTS message_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            message_id INTEGER,
            chat_id INTEGER,
            user_id INTEGER,
            nickname TEXT,
            message_text TEXT,
            model_score REAL,
            is_spam BOOLEAN,
            media_type TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)
        
        # Table for user activity
        await conn.execute("""
        CREATE TABLE IF NOT EXISTS user_activity (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            nickname TEXT,
            message_count INTEGER DEFAULT 1,
            first_seen TIMESTAMP,
            last_seen TIMESTAMP,
            is_active BOOLEAN DEFAULT FALSE,
            UNIQUE(user_id)
        );
        """)
        
        # Create indices for faster lookups
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_user_activity_user_id ON user_activity (user_id);")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_message_data_user_id ON message_data (user_id);")
        await conn.commit()

async def is_user_active(user_id):
    """
    Check if a user is considered active (has sent at least MSG_ACTIVE_USER_THRESHOLD messages in MSG_ACTIVE_USER_DAYS)
    """
    async with aiosqlite.connect(MSG_DB_PATH) as db:
        query = """
        SELECT is_active FROM user_activity WHERE user_id = ?;
        """
        async with db.execute(query, (user_id,)) as cursor:
            result = await cursor.fetchone()
            return result and result[0] if result else False

async def update_user_activity(user_id, nickname):
    """
    Update user activity in the database
    """
    now = datetime.datetime.now()
    
    async with aiosqlite.connect(MSG_DB_PATH) as db:
        # Check if user exists
        query = "SELECT message_count, first_seen FROM user_activity WHERE user_id = ?;"
        async with db.execute(query, (user_id,)) as cursor:
            result = await cursor.fetchone()
            
        if result:
            # User exists, update
            message_count = result[0] + 1
            is_active = message_count >= MSG_ACTIVE_USER_THRESHOLD
            
            await db.execute("""
                UPDATE user_activity 
                SET message_count = ?, last_seen = ?, nickname = ?, is_active = ?
                WHERE user_id = ?;
            """, (message_count, now, nickname, is_active, user_id))
        else:
            # New user
            await db.execute("""
                INSERT INTO user_activity (user_id, nickname, message_count, first_seen, last_seen, is_active)
                VALUES (?, ?, 1, ?, ?, FALSE);
            """, (user_id, nickname, now, now))
        
        await db.commit()

async def insert_message_data(message_id, chat_id, user_id, nickname, message_text, model_score, is_spam, media_type=None):
    """
    Insert message data into the database
    """
    async with aiosqlite.connect(MSG_DB_PATH) as db:
        await db.execute("""
            INSERT INTO message_data (message_id, chat_id, user_id, nickname, message_text, model_score, is_spam, media_type)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?);
        """, (message_id, chat_id, user_id, nickname, message_text, model_score, is_spam, media_type))
        await db.commit()

async def get_model_score(nickname, message_text):
    """
    Get the model score for a given message
    Format: {nickname} [SEP] {message}
    """
    # Format input
    input_text = f"{nickname} [SEP] {message_text}"
    
    # Tokenize
    encoded_input = tokenizer(
        [input_text], 
        padding=True, 
        truncation=True, 
        max_length=512, 
        return_tensors='pt'
    )
    input_ids = encoded_input['input_ids'].to(device)
    attention_mask = encoded_input['attention_mask'].to(device)

    # Prediction
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probs = torch.sigmoid(logits)[:, 1]
    
    return probs.cpu().numpy()[0]

async def handle_spam_message(event, message, model_score):
    """
    Handle a message detected as spam:
    1. Delete all messages from this user in the past 24 hours
    2. Ban the user (only once)
    3. Forward message to spam report channel
    """
    chat_id = message.chat_id
    message_id = message.id
    user_id = message.sender_id
    
    try:
        # Get chat information for reporting
        chat = await client.get_entity(chat_id)
        chat_name = f"@{chat.username}" if hasattr(chat, 'username') and chat.username else f"chat#{chat_id}"
        
        # Get user information
        sender = await client.get_entity(PeerUser(user_id))
        display_name = utils.get_display_name(sender)
        
        # Create profile link - format: tg://user?id={user_id}
        user_link = f"[{display_name}](tg://user?id={user_id})"
        
        # Check if user was already banned in the last hour (to prevent duplicate bans)
        ban_key = f"banned_{user_id}"
        if not hasattr(handle_spam_message, "banned_users"):
            handle_spam_message.banned_users = {}
        
        current_time = time.time()
        skip_ban = False
        if ban_key in handle_spam_message.banned_users:
            last_ban_time = handle_spam_message.banned_users[ban_key]
            # If banned in the last hour, skip the ban command
            if current_time - last_ban_time < 3600:  # 1 hour in seconds
                logging.info(f"User {user_id} was already banned recently, skipping ban command")
                skip_ban = True
        if not skip_ban:
            # First time banning this user
            await client.send_message(
                MSG_BAN_CHANNEL, 
                MSG_BAN_COMMAND.format(user_id=user_id, model_score=model_score)
            )
            handle_spam_message.banned_users[ban_key] = current_time
            try:
                wiki_game_chat = await client.get_entity('wikipedia_zh_game')
                # Workaround: ban directly due to global_ban api does not work there
                await client.edit_permissions(wiki_game_chat, user_id, view_messages=False)
            except Exception as ban_err:
                logging.error(f"Error banning user {user_id} on wiki_game_chat: {ban_err}")
        
        deleted_count = 0
        try:
            # Get all messages from this user in this chat
            async for msg in client.iter_messages(
                chat_id,
                from_user=user_id,
                limit=10,  # Reasonable limit to avoid too many API calls
            ):
                try:
                    await msg.delete()
                    deleted_count += 1
                except Exception as del_err:
                    logging.error(f"Error deleting message {msg.id}: {del_err}")
            
            logging.info(f"Deleted {deleted_count} messages from user {user_id} in the past 24 hours")
        except Exception as iter_err:
            logging.error(f"Error iterating messages: {iter_err}")
            # Still delete the current message if we can't delete all
            await event.delete()
        
        # Create message content for spam report
        report_text = f"{chat_name} 自动删除了以下来自 {user_link} (`{user_id}`) 的 spam 消息 (共 {deleted_count} 条消息), score={model_score:.4f}"
        
        # Add the original message content
        if message.message:
            # Truncate the message if it's too long (Telegram has 4096 char limit)
            msg_text = message.message
            if len(msg_text) > 3000:
                msg_text = msg_text[:3000] + "..."
            report_text += f"\n\n```\n{msg_text}\n```"
        else:
            report_text += f"\n\n[multimedia]"
        
        # Send report to the channel
        await client.send_message(
            MSG_SPAM_REPORT_CHANNEL,
            report_text,
            parse_mode='markdown'
        )
        
        logging.info(f"🚫 Spam message detected and actions taken for user {user_id} (score: {model_score:.4f})")
    except Exception as e:
        logging.error(f"Error handling spam message: {e}")

@client.on(events.Album(chats=MSG_GROUP_LIST))
async def handle_new_album(event):
    media_files = []
    chat = await event.get_chat()
    chat_id = chat.id
    sender = await event.get_sender()
    user_id = sender.id

    for msg in event.messages:
        file_path = await msg.download_media(file=bytes)
        media_files.append(file_path)

    logging.debug(f"Received album from {user_id} in {chat_id}, downloaded {len(media_files)} file(s)")
    if not media_files:
        return
    
    try:
        # To "encounter" the newly joined ID
        await client.get_participants(chat_id, limit=20)
        # Get user information
        sender = await client.get_entity(PeerUser(user_id))
        nickname = utils.get_display_name(sender)
        
        # Update user activity
        await update_user_activity(user_id, nickname)
        
        # Check if user is active - if yes, skip spam detection
        if await is_user_active(user_id):
            logging.debug(f"User {user_id} is active, skipping spam check")
            return
        
        chat_title = getattr(chat, 'title', f"Chat {event.chat_id}")
        sender_name = utils.get_display_name(sender)
        sender_id = sender.id
        
        logging.info(f"收到来自 {chat_title} ({event.chat_id}) 的媒体消息，发送者: {sender_name} ({sender_id})")
        
        for file in media_files:
            result = await classify_image(file, nsfw_model, feature_extractor)
            
            if "error" in result:
                logging.error(f"处理图像时出错: {result['error']}")
            else:
                logging.info(f"NSFW检测结果: {result['nsfw_probability']:.4f} ({'NSFW' if result['is_nsfw'] else '正常'})")
                
                if result['is_nsfw']:
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    logging.warning(f"检测到NSFW内容: 群组={chat_title}, 发送者={sender_name}, 概率={result['nsfw_probability']:.4f}")
                    
                    # Save the file to local for model training
                    save_dir = "nsfw_detected"
                    os.makedirs(save_dir, exist_ok=True)
                    with open(f"{save_dir}/nsfw_{timestamp}_{sender_id}.jpg", "wb") as f:
                        f.write(file)
                    
                    await handle_spam_message(event, event.messages[0], result['nsfw_probability'])

    except Exception as e:
        logging.error(f"Error processing album: {e}")
    

@client.on(events.NewMessage(chats=MSG_GROUP_LIST))
async def handle_new_message(event):
    """
    Handle new messages in monitored groups
    """
    if isinstance(event, events.Album):
        return
    message = event.message
    chat_id = event.chat_id
    user_id = message.sender_id
    
    # Skip messages from bots
    if message.sender:
        if hasattr(message.sender, 'bot') and message.sender.bot:
            return
    
    try:
        # To "encounter" the newly joined ID
        await client.get_participants(chat_id, limit=20)
        # Get user information
        sender = await client.get_entity(PeerUser(user_id))
        nickname = utils.get_display_name(sender)
        
        # Update user activity
        await update_user_activity(user_id, nickname)
        
        # Check if user is active - if yes, skip spam detection
        if await is_user_active(user_id):
            logging.debug(f"User {user_id} is active, skipping spam check")
            return
        
        # Get message text - handle regular text and media captions
        message_text = message.message
        media_type = None
        
        # Add quote text for detection if exists
        if hasattr(message, 'reply_to') and message.reply_to and hasattr(message.reply_to, 'quote_text') and message.reply_to.quote_text:
            quote_text = message.reply_to.quote_text
            message_text = f"{message_text}\n[QUOTE] {quote_text}"
            logging.debug(f"Message includes a quote: {quote_text[:100]}...")
        
        # Check for media content
        if message.media:
            media_type = str(type(message.media).__name__)
            
            # Check for photos/images
            if message.photo:
                file_path = await message.download_media(file=bytes)
                if file_path:
                    # Classify image using NSFW model
                    result = await classify_image(file_path, nsfw_model, feature_extractor)
                    
                    if "error" in result:
                        logging.error(f"Error processing image: {result['error']}")
                    else:
                        logging.info(f"NSFW detection result: {result['nsfw_probability']:.4f} ({'NSFW' if result['is_nsfw'] else 'Normal'})")
                        
                        if result['is_nsfw']:
                            # Save the file for model training
                            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                            save_dir = "nsfw_detected"
                            os.makedirs(save_dir, exist_ok=True)
                            with open(f"{save_dir}/nsfw_{timestamp}_{user_id}.jpg", "wb") as f:
                                f.write(file_path)
                            
                            # Handle spam message (delete and ban)
                            await handle_spam_message(event, message, result['nsfw_probability'])
                            return  # Skip text analysis if image is already detected as spam

        if not message_text:  # If no caption, skip the text check
            return
            
        # Get model score
        model_score = await get_model_score(nickname, message_text)
        # Naive bayes score
        nb_score = classifier.predict_proba([nickname + '[SEP]' + message_text])[0]
        
        # Log the score
        logging.info(f"Message from {nickname} ({user_id}) scored {model_score:.4f}; nb score {nb_score:.4f}")
        
        # Insert into database; require both model to reduce FP
        if len(message_text) < 10:
            is_spam = model_score >= MSG_SPAM_THRESHOLD and nb_score >= MSG_SPAM_THRESHOLD
        else:
            is_spam = (model_score >= MSG_SPAM_THRESHOLD and nb_score >= 0.06) or nb_score >= 0.94
        await insert_message_data(
            message.id, chat_id, user_id, nickname, 
            message_text, str(model_score), is_spam, media_type
        )
        
        # Handle spam
        if is_spam:
            logging.warning(f"⚠️ Detected spam message from {nickname} (score: {model_score:.4f}): {message_text[:100]}...")
            
            # For now, just print warning - uncomment to enable actual spam handling
            await handle_spam_message(event, message, max(model_score, nb_score))
            
    except Exception as e:
        logging.error(f"Error processing message: {e}")

@client.on(events.MessageEdited(chats=MSG_GROUP_LIST))
async def handle_edited_message(event):
    """
    Handle edited messages in monitored groups - similar logic to new messages
    """
    # Reuse most of the logic from handle_new_message
    await handle_new_message(event)

async def cleanup_old_data():
    """
    Periodically clean up old message data and recalculate user activity status
    """
    while True:
        try:
            cutoff_date = datetime.datetime.now() - datetime.timedelta(days=MSG_ACTIVE_USER_DAYS)
            
            async with aiosqlite.connect(MSG_DB_PATH) as db:
                # Delete old message data
                await db.execute(
                    "DELETE FROM message_data WHERE created_at < ?", 
                    (cutoff_date,)
                )
                
                # Recalculate active users based on message count within the active period
                await db.execute("""
                    UPDATE user_activity
                    SET is_active = (
                        SELECT CASE WHEN COUNT(*) >= ? THEN TRUE ELSE FALSE END
                        FROM message_data
                        WHERE user_id = user_activity.user_id
                          AND created_at > ?
                    )
                """, (MSG_ACTIVE_USER_THRESHOLD, cutoff_date))
                
                await db.commit()
                
            logging.info("Cleaned up old data and updated user activity status")
        except Exception as e:
            logging.error(f"Error during data cleanup: {e}")
        
        # Run once a day
        await asyncio.sleep(86400)

async def main():
    """
    Main function to run the message spam detector
    """
    await init_db()
    
    # Start periodic cleanup
    asyncio.create_task(cleanup_old_data())
    
    # Connect and run the client
    await client.start()
    
    # Log start
    logging.info(f"Message spam detector started. Running on {device}.")
    logging.info(f"Monitoring {len(MSG_GROUP_LIST)} groups for spam messages.")
    
    # Keep the client running
    await client.run_until_disconnected()

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Message spam detector stopped by user")
        exit(0)
