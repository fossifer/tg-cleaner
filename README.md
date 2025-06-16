# Telegram Spam Killer

A Telegram bot that uses machine learning to detect and handle spam messages in Telegram groups. The bot uses both text and image analysis to identify spam and NSFW content.

## Features

- Text-based spam detection using DistilBERT and Naive Bayes
- Image-based NSFW content detection using ViT
- User profile spam detection using DistilBERT
- Automatic message deletion and user banning
- User activity tracking to reduce false positives
- Spam report channel integration
- Support for multiple groups

## Components

The project consists of two main components:

1. `infer_msgs.py`: Message spam detection
   - Monitors messages in configured groups
   - Detects spam messages using text analysis
   - Detects NSFW content in images
   - Deletes spam messages and bans users

2. `infer_profile.py`: User profile spam detection
   - Monitors new user joins in configured groups
   - Analyzes user profiles (nickname, username, bio)
   - Bans suspicious users before they can spam

## Setup

1. Clone the repository:
```bash
git clone https://github.com/fossifer/tg-spam-killer.git
cd tg-spam-killer
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your configuration:
   - Copy `config.example.py` to `config.py`
   - Fill in your Telegram API credentials (get them from https://my.telegram.org)
   - Configure your group list and channels
   - Adjust thresholds if needed

4. Download the required models:
   - Place your trained models in the `models/` directory
   - Required models:
     - `distilbert-messages_best_model.pt` (for message detection)
     - `distilbert_concat_best_model.pt` (for profile detection)
     - `vit-nsfw-custom` (for NSFW detection)
     - `nb_model_and_vectorizer.joblib` (for message detection)

5. Run the bot:
```bash
# Run both components
python run.py

# Or run components separately
python infer_msgs.py  # Message spam detection
python infer_profile.py  # Profile spam detection
```

## Configuration

Edit `config.py` to customize:
- API credentials
- Spam and NSFW thresholds
- Group lists
- Report channels
- Ban commands

## Security

- Never commit your `config.py` file
- Keep your API credentials secure
- The `.gitignore` file is set up to prevent sensitive files from being uploaded

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 