#!/usr/bin/env sh
set -e

# Defaults (can be overridden via env or flags)
IMAGE_NAME=${IMAGE_NAME:-postgres:18.1}
CONTAINER_NAME=${CONTAINER_NAME:-postgres-wwj}
PORT_MAP=${PORT_MAP:-12345:5432}
TZ=${TZ:-Asia/Shanghai}
POSTGRES_DATA_DIR=${POSTGRES_DATA_DIR:-$PWD/pg_data}
POSTGRES_USER=${POSTGRES_USER:-appuser}
POSTGRES_PASSWORD=${POSTGRES_PASSWORD:-apppassword}
POSTGRES_DB=${POSTGRES_DB:-postgres}
INIT_DIR=${INIT_DIR:-}
FILTER_PREFIX=${FILTER_PREFIX:-}
USER_SPECIFIED_FILTER=false
USER_SPECIFIED_INIT_DIR=false

print_usage() {
  echo "Usage: $0 [-i image_name] [-c container_name] [-m port_map] [-d data_dir] [-u user] [-p password] [-b db_name] [-s init_dir] [-f filter_prefix]";
  echo "  -i image_name      Docker image tag (default: ${IMAGE_NAME})";
  echo "  -c container_name  Container name (default: ${CONTAINER_NAME})";
  echo "  -m port_map        Port mapping (default: ${PORT_MAP})";
  echo "  -t time_zone       Timezone (default: ${TZ})";
  echo "  -d data_dir        Host data directory for PGDATA (default: ${POSTGRES_DATA_DIR})";
  echo "  -u user            Postgres user (default: ${POSTGRES_USER})";
  echo "  -p password        Postgres password (default: ${POSTGRES_PASSWORD})";
  echo "  -b db_name         Default database name (default: ${POSTGRES_DB})";
  echo "  -s init_dir        Host dir with init scripts, mounted to /docker-entrypoint-initdb.d (optional)";
  echo "  -f filter_prefix   File prefix filter (applies to auto-detected or -s directory, if specified)";
}

while getopts "i:c:m:t:d:u:p:b:s:f:h" opt; do
  case "$opt" in
    i) IMAGE_NAME="$OPTARG" ;;
    c) CONTAINER_NAME="$OPTARG" ;;
    m) PORT_MAP="$OPTARG" ;;
    t) TZ="$OPTARG" ;;
    d) POSTGRES_DATA_DIR="$OPTARG" ;;
    u) POSTGRES_USER="$OPTARG" ;;
    p) POSTGRES_PASSWORD="$OPTARG" ;;
    b) POSTGRES_DB="$OPTARG" ;;
    s) INIT_DIR="$OPTARG"; USER_SPECIFIED_INIT_DIR=true ;;
    f) FILTER_PREFIX="$OPTARG"; USER_SPECIFIED_FILTER=true ;;
    h) print_usage; exit 0 ;;
    *) print_usage; exit 1 ;;
  esac
done

# If INIT_DIR not set, but docker/docker-init.d exists and is not empty, use it by default
if [ -z "$INIT_DIR" ]; then
  DEFAULT_INIT_DIR="$(dirname "$0")/docker-init.d"
  if [ -d "$DEFAULT_INIT_DIR" ] && [ "$(ls -A "$DEFAULT_INIT_DIR")" ]; then
    INIT_DIR="$DEFAULT_INIT_DIR"
    # Set default filter prefix only if user didn't specify -f
    if [ "$USER_SPECIFIED_FILTER" = false ]; then
      FILTER_PREFIX="postgres"
    fi
    echo "Auto-detected init dir: $INIT_DIR (files with prefix: $FILTER_PREFIX)"
  fi
fi

echo "========================================";
echo "Image:        $IMAGE_NAME";
echo "Container:    $CONTAINER_NAME";
echo "Port map:     $PORT_MAP";
echo "Time Zone:    $TZ";
echo "Data dir:     $POSTGRES_DATA_DIR";
echo "DB:           $POSTGRES_DB";
echo "User:         $POSTGRES_USER";
[ -n "$POSTGRES_PASSWORD" ] && echo "Password:     *******";
[ -n "$INIT_DIR" ] && echo "Init dir:     $INIT_DIR";
[ -n "$INIT_DIR" ] && [ -n "$FILTER_PREFIX" ] && echo "File prefix:  $FILTER_PREFIX*";
echo "========================================";

if ! command -v docker >/dev/null 2>&1; then
  echo "Error: docker not found in PATH" >&2;
  exit 1;
fi

mkdir -p "$POSTGRES_DATA_DIR";
[ -n "$INIT_DIR" ] && mkdir -p "$INIT_DIR";

# Check if Dockerfile.postgres exists
if [ ! -f "Dockerfile.postgres" ]; then
  echo "Error: Dockerfile.postgres not found in current directory" >&2;
  exit 1;
fi

echo "Building image $IMAGE_NAME ...";
docker build -f Dockerfile.postgres -t "$IMAGE_NAME" .;

if docker ps -a --format '{{.Names}}' | grep -w "$CONTAINER_NAME" >/dev/null 2>&1; then
  echo "Stopping existing container $CONTAINER_NAME (if running) ...";
  if docker ps --format '{{.Names}}' | grep -w "$CONTAINER_NAME" >/dev/null 2>&1; then
    docker stop "$CONTAINER_NAME" || true;
  fi
  echo "Removing existing container $CONTAINER_NAME ...";
  docker rm -f "$CONTAINER_NAME" || true;
fi

# Create a temporary init directory if filtering is needed
# Filter when:
# 1. Auto-detected mode (no -s specified): always apply filter (use default or user-specified prefix)
# 2. Manual mode with -s: only filter if user explicitly specified -f
TEMP_INIT_DIR=""
SHOULD_FILTER=false

if [ "$USER_SPECIFIED_INIT_DIR" = false ]; then
  # Auto-detected mode: always filter
  SHOULD_FILTER=true
elif [ "$USER_SPECIFIED_FILTER" = true ]; then
  # Manual mode with -s: filter only if -f was explicitly specified
  SHOULD_FILTER=true
fi

if [ "$SHOULD_FILTER" = true ] && [ -n "$INIT_DIR" ] && [ -n "$FILTER_PREFIX" ]; then
  TEMP_INIT_DIR=$(mktemp -d)
  echo "Creating filtered init directory: $TEMP_INIT_DIR (copying files with prefix: $FILTER_PREFIX)"
  for file in "$INIT_DIR"/$FILTER_PREFIX*; do
    if [ -f "$file" ]; then
      cp "$file" "$TEMP_INIT_DIR/"
      echo "  Copied: $(basename "$file")"
    fi
  done
  # Only use temp dir if we actually copied files
  if [ "$(ls -A "$TEMP_INIT_DIR")" ]; then
    INIT_DIR="$TEMP_INIT_DIR"
  else
    echo "Warning: No files found with prefix '$FILTER_PREFIX' in $INIT_DIR"
    rm -rf "$TEMP_INIT_DIR"
    TEMP_INIT_DIR=""
  fi
fi

echo "Starting new container $CONTAINER_NAME ...";
set -- \
  --name "$CONTAINER_NAME" \
  -p "$PORT_MAP" \
  -e POSTGRES_USER="$POSTGRES_USER" \
  -e POSTGRES_PASSWORD="$POSTGRES_PASSWORD" \
  -e POSTGRES_DB="$POSTGRES_DB" \
  -e TZ="$TZ" \
  -v "$POSTGRES_DATA_DIR":/var/lib/postgresql

if [ -n "$INIT_DIR" ]; then
  set -- "$@" -v "$INIT_DIR":/docker-entrypoint-initdb.d
fi

docker run -d --restart=unless-stopped "$@" "$IMAGE_NAME";

# Cleanup temp directory if it was created
if [ -n "$TEMP_INIT_DIR" ]; then
  rm -rf "$TEMP_INIT_DIR"
fi

echo "Container started. Showing last 30 log lines (Ctrl-C to exit):";
docker logs -f -t --tail 30 "$CONTAINER_NAME";

# sudo ./docker-postgres.sh -s ./docker-init.d -f postgres