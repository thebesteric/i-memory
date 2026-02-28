#!/bin/bash

# 脚本说明：
# 该脚本用于启动应用程序。
# 它会检查环境模式参数是否有效，激活 Conda 环境，设置环境变量，
# 并在后台运行应用程序，将输出重定向到日志文件。
# 常见问题：
# 1. bash: ./run.sh: /bin/bash^M: bad interpreter: No such file or directory
# 解决方法：使用 dos2unix 工具将 run.sh 转换为 unix 格式
# 安装：sudo apt install dos2unix
# 命令：dos2unix run.sh

# 配置项
APP_NAME="i-memory"
CONDA_ENV="i-memory"
RUN_SCRIPT="src/web/api.py"
LOG_FILE="i-memory.log"
PID_FILE="i-memory.pid"
# Python 解释器路径（根据实际情况修改，可通过 which python3 查看）
PYTHON_CMD="python3"
# 允许的环境模式
ALLOWED_MODES=("local" "dev" "test" "prod")

# 检查环境模式是否有效
check_env_mode() {
    local mode="$1"
    for allowed in "${ALLOWED_MODES[@]}"; do
        if [ "$mode" = "$allowed" ]; then
            return 0  # 模式有效
        fi
    done
    echo "错误：环境模式必须是以下之一：${ALLOWED_MODES[*]}"
    # 模式无效
    return 1
}

# 启动程序
start() {
    if [ -f "$PID_FILE" ]; then
        echo "程序已在运行中，PID: $(cat $PID_FILE)"
        return 0
    fi

    # 检查是否提供了环境模式参数
    if [ -z "$1" ]; then
        echo "请指定环境模式，使用方式: $0 start <env-mode>"
        echo "允许的模式：${ALLOWED_MODES[*]}"
        return 1
    fi

    # 检查环境模式有效性
    if ! check_env_mode "$1"; then
        return 1
    fi
    # 刷新环境变量
    source ~/.bashrc
    # 获取用户指定的环境模式参数
    local env_mode="$1"
    # 设置环境变量
    export ENVIRONMENT=$env_mode
    # 激活 Conda 环境
    source activate $CONDA_ENV
    echo "正在启动 $APP_NAME (应用环境: $env_mode，虚拟环境: $CONDA_ENV) ..."
    # 后台运行并将输出重定向到日志文件，添加环境模式参数
    nohup $PYTHON_CMD $RUN_SCRIPT --env-mode="$env_mode" > $LOG_FILE 2>&1 &
    # 记录进程ID
    echo $! > $PID_FILE

    # 检查是否启动成功
    sleep 1
    if [ -f "$PID_FILE" ] && ps -p $(cat $PID_FILE) > /dev/null; then
        echo "启动成功，PID: $(cat $PID_FILE)"
        echo "日志文件: $LOG_FILE"
    else
        echo "启动失败，请查看日志文件"
        rm -f $PID_FILE
    fi
}

# 停止程序
stop() {
    if [ ! -f "$PID_FILE" ]; then
        echo "程序未在运行"
        return 0
    fi

    PID=$(cat $PID_FILE)
    echo "停止 PID: $PID ..."

    # 尝试正常终止
    kill $PID
    sleep 2

    # 若仍在运行则强制终止
    if ps -p $PID > /dev/null; then
        echo "强制终止程序..."
        kill -9 $PID
        sleep 1
    fi

    rm -f $PID_FILE
    echo "程序已停止"
}

# 重启程序
restart() {
    # 检查是否提供了环境模式参数
    if [ -z "$1" ]; then
        echo "请指定环境模式，使用方式: $0 restart <env-mode>"
        echo "允许的模式：${ALLOWED_MODES[*]}"
        return 1
    fi

    # 检查环境模式有效性
    if ! check_env_mode "$1"; then
        return 1
    fi

    stop
    start "$1"
}

# 查看状态
status() {
    if [ -f "$PID_FILE" ] && ps -p $(cat $PID_FILE) > /dev/null; then
        echo "$APP_NAME 正在运行，PID: $(cat $PID_FILE)"
    else
        echo "$APP_NAME 未在运行"
        rm -f $PID_FILE  # 清理无效的PID文件
    fi
}

# 查看日志
logs() {
    if [ -f "$LOG_FILE" ]; then
        echo "查看 $LOG_FILE 最新内容（按 Ctrl+C 退出）："
        tail -f $LOG_FILE
    else
        echo "日志文件不存在: $LOG_FILE"
    fi
}

# 帮助信息
usage() {
    echo "使用方法: $0 {start | stop | restart | status | logs} [env-mode]"
    echo "  start   - 启动程序，需指定环境模式: $0 start <env-mode>"
    echo "  stop    - 停止程序"
    echo "  restart - 重启程序，需指定环境模式: $0 restart <env-mode>"
    echo "  status  - 查看程序状态"
    echo "  logs    - 查看实时日志"
    echo "允许的环境模式：${ALLOWED_MODES[*]}"
    exit 1
}

# 解析参数
case "$1" in
    start)
        start "$2"
        ;;
    stop)
        stop
        ;;
    restart)
        restart "$2"
        ;;
    status)
        status
        ;;
    logs)
        logs
        ;;
    *)
        usage
        ;;
esac