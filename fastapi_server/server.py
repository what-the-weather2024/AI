import importlib
import os

import uvicorn
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware


class FastAPIServer:
    def __init__(self):
        self.app = FastAPI(
            title="What the Weather AI Module Backend",
        )
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=[
                org.strip() for org in os.getenv("CORS_ORIGINS", "").split(",")
            ],
            allow_credentials=False,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        @self.app.get("/")
        async def index():
            return "AutoTrade API running ..."

        @self.app.get("/health")
        async def health():
            return {"code": 200}

    def run(self):
        uvicorn.run(self.app, host="127.0.0.1", port=8000)


def get_controllers(modules):
    return [importlib.import_module(module_name) for module_name in modules]


def ensure_directory_exists(save_path: str):
    # 부모 디렉토리 경로를 가져옵니다.
    parent_directory = os.path.dirname(save_path)

    # 부모 디렉토리가 존재하는지 확인하고, 없으면 생성합니다.
    if not os.path.exists(parent_directory):
        os.makedirs(parent_directory)
