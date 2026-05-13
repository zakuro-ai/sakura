import secrets
from datetime import datetime


def new_job_id(prefix: str = "job") -> str:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    random_hex = secrets.token_hex(4)
    return f"{prefix}-{timestamp}-{random_hex}"
