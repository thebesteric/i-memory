const path = require('path')

const projectRoot = path.resolve(__dirname, '.')
const logsDir = path.join(projectRoot, 'logs')

module.exports = {
    apps: [
        {
            name: 'i-memory-backend',
            cwd: projectRoot,
            script: path.join(projectRoot, '.venv/bin/python'),
            args: '-m uvicorn interfaces.api.app:create_app --app-dir src --factory --host 0.0.0.0 --port 18088',
            interpreter: 'none',
            env: {
                PYTHONPATH: 'src',
            },
            out_file: path.join(logsDir, 'i-memory-out.log'),
            error_file: path.join(logsDir, 'i-memory-error.log'),
            log_date_format: 'YYYY-MM-DD HH:mm:ss Z',
            autorestart: true,
            watch: false,
            max_memory_restart: '8G',
        },
    ],
}
