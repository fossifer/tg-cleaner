# Telegram API credentials
BOT_TOKEN = '1234567890:Abcdefghijklmnopqrstuvwxyz'  # Replace with your bot token
API_URL = 'https://api.telegram.org/bot' + BOT_TOKEN + '/'
API_ID = 123456  # Replace with your API ID
API_HASH = 'your_api_hash_here'  # Replace with your API hash

# Common settings
SEED = 42  # Random seed for reproducibility

# Message spam detection (infer_msgs.py) settings
MSG_DB_PATH = "message_data.db"
MSG_MODEL_PATH = "models/distilbert-messages_best_model.pt"
MSG_SPAM_THRESHOLD = 0.8  # For text
MSG_NSFW_THRESHOLD = 0.7  # For image
MSG_ACTIVE_USER_THRESHOLD = 3  # Number of messages to consider a user active
MSG_ACTIVE_USER_DAYS = 180  # Number of days to check for active users

# Message spam detection channels and groups
MSG_SPAM_REPORT_CHANNEL = 1234567890  # Channel to forward spam messages to
MSG_BAN_CHANNEL = -1001234567890  # Channel to send ban commands
MSG_BAN_COMMAND = '/gbb {user_id} -r "自动判断为spam消息, score={model_score:.4f}\n 误判? 在群管群发送 `/wl {user_id}`"'

# User profile spam detection (tgmain.py) settings
USER_DB_PATH = "user_data.db"
USER_MODEL_PATH = 'models/distilbert_concat_best_model.pt'
USER_SPAM_THRESHOLD = 0.85  # Threshold for user profile spam detection

# User profile spam detection channels and groups
USER_BAN_CHANNEL = -1001234567890  # Channel to send ban commands
USER_BAN_COMMAND = '/gbb {user_id} -r "自动判断为spam, score={model_score}\n 误判? 在群管群发送 `/wl {user_id}`"'
USER_UNBAN_COMMAND = '/global_unban {user_id} -r "被群管标记为误判的模型自动封锁"'
USER_CAPTCHA_BOT_ID = 123456789  # If set, will also delete corresponding captcha messages when banning a user

# Model paths
NSFW_MODEL_PATH = "models/vit-nsfw-custom"

# Channels and groups
SPAM_REPORT_CHANNEL = 1234567890  # Replace with your spam report channel ID
BAN_CHANNEL = -1001234567890  # Replace with your ban channel ID
BAN_COMMAND = '/gbb {user_id} -r "自动判断为spam消息, score={model_score:.4f}\n 误判? 在群管群发送 `/wl {user_id}`"'

# Groups to monitor - add your group usernames or IDs here
GROUP_LIST = {
    'group1',
    'group2',
    # Add more groups as needed
} 

# Message spam detection groups to monitor
MSG_GROUP_LIST = GROUP_LIST

# User profile spam detection groups to monitor
USER_GROUP_LIST = GROUP_LIST
