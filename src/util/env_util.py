from dotenv import load_dotenv
from pathlib import Path
import logging

logger = logging.getLogger("PULSE_logger")


def load_environment():
    """
    Load Api URI's and Key as environment variables.
    Place .env file into secrets folder. Make sure that api_key_name and api_uri_name match to model config.
    """
    env_path = Path(__file__).resolve().parents[2] / "secrets" / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
        logger.info("Environment variables loaded from secrets/.env")
    else:
        logger.info("secrets/.env file not found.")
