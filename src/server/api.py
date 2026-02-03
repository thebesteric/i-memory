from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import time
from .routes import memory, health, sources
from ..core.config import EnvConfig
from ..module import app_injector
from ..utils.log_helper import LogHelper

logger = LogHelper.get_logger()

env: EnvConfig = app_injector.get(EnvConfig)


def create_app() -> FastAPI:
    app = FastAPI(title="iMemory API", version="1.0.0")

    # CORS Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        start = time.time()
        response = await call_next(request)
        process_time = (time.time() - start) * 1000
        logger.info(f"{request.method} {request.url.path} - {response.status_code} ({process_time:.2f}ms)")
        return response

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """
        FastAPI 生命周期管理
        """
        # 启动阶段：执行初始化操作
        logger.info(f"🚀 iMemory Server running on port {env.port}")

        yield  # 应用运行阶段

        # 关闭阶段：执行资源释放操作
        logger.info("🛑 iMemory Server shutting down...")

    # 注册路由
    app.include_router(health.router)
    app.include_router(memory.router)
    app.include_router(sources.router)

    return app
