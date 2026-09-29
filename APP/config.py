from dotenv import load_dotenv
import os

load_dotenv()

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")

AWS_SECRET_ACCESS_KEY = os.getenv(
    "AWS_SECRET_ACCESS_KEY"
)

AWS_SESSION_TOKEN = os.getenv(
    "AWS_SESSION_TOKEN"
)

AWS_REGION = os.getenv("AWS_REGION")

ATHENA_DATABASE = os.getenv(
    "ATHENA_DATABASE"
)

ATHENA_OUTPUT_LOCATION = os.getenv(
    "ATHENA_OUTPUT_LOCATION"
)