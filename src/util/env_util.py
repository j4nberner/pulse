import logging
from pathlib import Path

from dotenv import load_dotenv

logger = logging.getLogger("PULSE_logger")


def load_environment(env_file=None):
        """
        Load Api URI's and Key as environment variables.
        Place .env file into secrets folder. Make sure that api_key_name and api_uri_name match to model config.
        """
        if env_file is not None:
            env_path = Path(env_file)
            load_dotenv(dotenv_path=env_path)
        else:
            env_path = Path(__file__).resolve().parents[2] / "secrets" / ".env"
            load_dotenv(dotenv_path=env_path)