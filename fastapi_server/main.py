import os

from fastapi_server.initializer import Initializer
from fastapi_server.utils import FastAPIServer, get_controllers

Initializer()

server = FastAPIServer()
app = server.app

modules = get_controllers(
    [
        f"fastapi_server.controllers.{name.strip().replace('.py', '')}"
        for name in os.listdir("./fastapi_server/controllers")
        if "controller.py" in name
    ]
)

for module in modules:
    router = module.router
    app.include_router(router)

server.run()
