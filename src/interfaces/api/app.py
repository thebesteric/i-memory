import os
import inspect
import sys
import time
from contextlib import asynccontextmanager

import warnings
# 屏蔽正则转义语法警告
warnings.filterwarnings("ignore", category=SyntaxWarning)
# 屏蔽 pkg_resources 弃用警告
warnings.filterwarnings("ignore", category=UserWarning, module="jieba")

from agile.commons.biz_error import BizError
from agile.utils import LogHelper
from agile.web import R
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse

# 获取当前脚本（api.py）的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录（i-memory），根据你的目录结构调整 ../ 的层数
project_root = os.path.abspath(os.path.join(current_dir, '../../../'))
# 将项目根目录加入 Python 搜索路径
sys.path.append(project_root)

from shared.config.settings import env
from services.memory.components import get_vector_store
from infra.db.orm_models import init_db_schema
from infra.scheduler.jobs import start_background_tasks, stop_background_tasks
from interfaces.api.routes import health_router, backend_router, auth_router, graph_router, memory_router

logger = LogHelper.get_logger()


def create_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """
        FastAPI 生命周期管理
        :param app: FastAPI
        """
        # 启动阶段：执行初始化操作
        env_mode = os.getenv("ENV_MODE")
        debug = True if env_mode and env_mode in ["", "dev"] else False
        logger.info(
            f"🚀 Starting iMemory API server on {env.WEB_HOST}:{env.WEB_PORT} with debug: {debug}, "
            f"using environment mode: {env_mode if env_mode else 'local'}"
        )

        # 初始化数据库
        await init_db_schema()
        logger.info("Database schema initialization completed")

        vector_store = get_vector_store()
        warmup = getattr(vector_store, "warmup", None)
        if callable(warmup):
            warmup_result = warmup()
            if inspect.isawaitable(warmup_result):
                await warmup_result
            logger.info("Vector store warmup completed")

        # 启动相关任务
        app.state.scheduler = start_background_tasks()

        yield  # 应用运行阶段

        # 关闭阶段：执行资源释放操作
        await stop_background_tasks(getattr(app.state, "scheduler", None))
        logger.info("🛑 iMemory Server shutting down...")

    app = FastAPI(
        title="iMemory API",
        version="1.0.0",
        lifespan=lifespan,
        docs_url=None,
        redoc_url="/redoc"
    )

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

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        path = str(request.url)
        path_dict = {"path": path}
        # 处理 HTTP 异常
        if isinstance(exc, HTTPException):
            logger.error(f"HTTP 异常信息捕获: Path={request.url}, Status={exc.status_code}, Detail={exc.detail}")
            return JSONResponse(
                status_code=exc.status_code,
                content=R.error().set_code(exc.status_code).set_message(str(exc.detail)).set_data(
                    path_dict).model_dump()
            )
        # 处理请求验证异常
        elif isinstance(exc, RequestValidationError):
            logger.error(f"请求验证异常信息捕获: Path={request.url}, Exception={str(exc)}")
            return JSONResponse(
                status_code=422,
                content=R.error().set_code(422).set_message(f"请求参数验证失败: {str(exc)}").set_data(
                    path_dict).model_dump()
            )
        elif isinstance(exc, BizError):
            logger.error(f"业务异常信息捕获: Path={request.url}, Exception={str(exc)}")
            data = exc.data
            data.update(path_dict)
            data.update({"module": exc.module})
            code = int(exc.code) if isinstance(exc.code, (int, str)) else 422
            return JSONResponse(
                status_code=422,
                content=R.error().set_code(code).set_message(f"业务处理异常: {str(exc)}").set_data(data).model_dump()
            )
        # 处理其他异常
        else:
            logger.error(f"服务器内部异常信息捕获: Path={request.url}, Exception={str(exc)}", exc_info=True)
            return JSONResponse(
                status_code=500,
                content=R.error().set_code(500).set_message(f"服务器内部错误: {str(exc)}").set_data(
                    path_dict).model_dump()
            )

    # 注册路由
    api_prefix = "/imemory"
    routers = [
        health_router.router,
        memory_router.router,
        graph_router.router,
        auth_router.router,
        backend_router.router,
    ]
    for router in routers:
        app.include_router(router, prefix=api_prefix)

    return app


if __name__ == '__main__':
    import uvicorn

    debug = env.WEB_DEBUG if env.WEB_DEBUG is not None else False
    uvicorn.run("src.interfaces.api.app:create_app", host=env.WEB_HOST, port=env.WEB_PORT, reload=debug)
