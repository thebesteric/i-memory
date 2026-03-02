#!/usr/bin/env sh
set -e

# 默认参数（可通过环境变量或命令行覆盖）
IMAGE_NAME=${IMAGE_NAME:-redis:8.4}
CONTAINER_NAME=${CONTAINER_NAME:-redis-wwj}
PORT_MAP=${PORT_MAP:-16379:6379}
REDIS_DATA_DIR=${REDIS_DATA_DIR:-$PWD/redis_data}
REDIS_PASSWORD=${REDIS_PASSWORD:-apppassword}
REDIS_CONF=${REDIS_CONF:-$PWD/redis.conf}  # 新增配置文件参数，默认当前目录

print_usage() {
  echo "Usage: $0 [-i image_name] [-c container_name] [-m port_map] [-d data_dir] [-f redis_conf] [-a redis_password]";
  echo "  -i image_name      Docker image tag (default: ${IMAGE_NAME})";
  echo "  -c container_name  Container name (default: ${CONTAINER_NAME})";
  echo "  -m port_map        Port mapping (default: ${PORT_MAP})";
  echo "  -d data_dir        Host data directory for Redis (default: ${REDIS_DATA_DIR})";
  echo "  -f redis_conf      Path to redis.conf file (default: ${REDIS_CONF})";
  echo "  -a password        Redis password (default: not set)";
}

while getopts "i:c:m:d:f:a:h" opt; do
  case "$opt" in
    i) IMAGE_NAME="$OPTARG" ;;
    c) CONTAINER_NAME="$OPTARG" ;;
    m) PORT_MAP="$OPTARG" ;;
    d) REDIS_DATA_DIR="$OPTARG" ;;
    f) REDIS_CONF="$OPTARG" ;;
    a) REDIS_PASSWORD="$OPTARG" ;;
    h) print_usage; exit 0 ;;
    *) print_usage; exit 1 ;;
  esac
  done


echo "========================================";
echo "Image:        $IMAGE_NAME";
echo "Container:    $CONTAINER_NAME";
echo "Port map:     $PORT_MAP";
echo "Data dir:     $REDIS_DATA_DIR";
if [ -f "$REDIS_CONF" ]; then
  echo "Redis conf:   $REDIS_CONF (will be mounted)";
  echo "Notice:      redis.conf 存在，将以配置文件启动，环境变量设置的密码不会生效";
else
  [ -n "$REDIS_PASSWORD" ] && echo "Password:     ******* (set by env)";
  echo "Notice:      未挂载 redis.conf，将使用环境变量配置 Redis 密码";
fi
echo "========================================";

if ! command -v docker >/dev/null 2>&1; then
  echo "Error: docker not found in PATH" >&2;
  exit 1;
fi

mkdir -p "$REDIS_DATA_DIR";

# 检查 Dockerfile.redis 是否存在
if [ ! -f "Dockerfile.redis" ]; then
  echo "Error: Dockerfile.redis not found in current directory" >&2;
  exit 1;
fi

echo "Building image $IMAGE_NAME ...";
docker build -f Dockerfile.redis -t "$IMAGE_NAME" .;

if docker ps -a --format '{{.Names}}' | grep -w "$CONTAINER_NAME" >/dev/null 2>&1; then
  echo "Stopping existing container $CONTAINER_NAME (if running) ...";
  if docker ps --format '{{.Names}}' | grep -w "$CONTAINER_NAME" >/dev/null 2>&1; then
    docker stop "$CONTAINER_NAME" || true;
  fi
  echo "Removing existing container $CONTAINER_NAME ...";
  docker rm -f "$CONTAINER_NAME" || true;
fi

echo "Starting new container $CONTAINER_NAME ...";
if [ -f "$REDIS_CONF" ]; then
  docker run -d \
    --restart=unless-stopped \
    --name "$CONTAINER_NAME" \
    -p "$PORT_MAP" \
    -v "$REDIS_DATA_DIR":/data \
    -v "$REDIS_CONF":/etc/redis/redis.conf \
    "$IMAGE_NAME" redis-server /etc/redis/redis.conf
else
  docker run -d \
    --restart=unless-stopped \
    --name "$CONTAINER_NAME" \
    -p "$PORT_MAP" \
    -v "$REDIS_DATA_DIR":/data \
    ${REDIS_PASSWORD:+-e REDIS_PASSWORD="$REDIS_PASSWORD"} \
    "$IMAGE_NAME"
fi

echo "Container started. Showing last 30 log lines (Ctrl-C to exit):";
docker logs -f -t --tail 30 "$CONTAINER_NAME";
