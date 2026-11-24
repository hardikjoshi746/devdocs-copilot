import jwt
import time
import os
from dotenv import load_dotenv

load_dotenv()

token = jwt.encode(
    {
        "sub": "hardik",
        "repos": ["job_scrapper", "ShopSphere"],
        "exp": int(time.time()) + 86400,
    },
    os.environ["JWT_SIGNING_SECRET"],
    algorithm="HS256",
)
print(token)