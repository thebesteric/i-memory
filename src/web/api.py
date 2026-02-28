import argparse
import os
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import time

from starlette.responses import JSONResponse
from utils.env_helper import EnvHelper
from utils.log_helper import LogHelper
from web.common_result import R

# 获取当前脚本（api.py）的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录（i-memory），根据你的目录结构调整 ../ 的层数
project_root = os.path.abspath(os.path.join(current_dir, '../../'))
# 将项目根目录加入 Python 搜索路径
sys.path.append(project_root)

from src.core.config import env
from src.web.routes import health_router, memory_router, sources_router

logger = LogHelper.get_logger()
env_helper = EnvHelper()


def create_app() -> FastAPI:
    app = FastAPI(title="iMemory API", version="1.0.0")

    # CORS Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
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
        :param app: FastAPI
        """
        # 启动阶段：执行初始化操作
        logger.info(f"🚀 iMemory Server running on port {env.WEB_PORT}")

        yield  # 应用运行阶段

        # 关闭阶段：执行资源释放操作
        logger.info("🛑 iMemory Server shutting down...")

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        path = str(request.url)
        # 处理 HTTP 异常
        if isinstance(exc, HTTPException):
            logger.error(f"HTTP 异常信息捕获: Path={request.url}, Status={exc.status_code}, Detail={exc.detail}")
            return JSONResponse(
                status_code=exc.status_code,
                content=R.error().set_code(exc.status_code).set_message(str(exc.detail)).set_data({"path": path}).model_dump()
            )
        # 处理请求验证异常
        elif isinstance(exc, RequestValidationError):
            logger.error(f"请求验证异常信息捕获: Path={request.url}, Exception={str(exc)}")
            return JSONResponse(
                status_code=422,
                content=R.error().set_code(422).set_message(f"请求参数验证失败: {str(exc)}").set_data({"path": path}).model_dump()
            )
        # 处理其他异常
        else:
            logger.error(f"服务器内部异常信息捕获: Path={request.url}, Exception={str(exc)}", exc_info=True)
            return JSONResponse(
                status_code=500,
                content=R.error().set_code(500).set_message(f"服务器内部错误: {str(exc)}").set_data({"path": path}).model_dump()
            )

    # 注册路由
    api_prefix = "/imemory"
    routers = [
        health_router.router,
        memory_router.router,
        sources_router.router
    ]
    for router in routers:
        app.include_router(router, prefix=api_prefix)

    return app


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-mode',
                        dest='env_mode',
                        type=str,
                        default='',
                        help='设置运行环境，可选值：dev/test/prod')
    args = parser.parse_args()
    env_mode = args.env_mode if args.env_mode else ""
    env_helper.set("ENV_MODE", env_mode)

    import uvicorn

    debug = True if env_mode in ["", "dev"] else False
    logger.info(f"Starting iMemory API server on {env.WEB_HOST}:{env.WEB_PORT} with debug={debug}, using environment mode: {env_mode if env_mode else "local"}")

    uvicorn.run("src.web.api:create_app", host=env.WEB_HOST, port=env.WEB_PORT, reload=debug)
