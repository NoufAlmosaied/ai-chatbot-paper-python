"""
Gunicorn configuration for PhishGuard AI Chatbot
Optimized for production deployment on Render.com
"""

import os
import multiprocessing

# Server socket
bind = f"0.0.0.0:{os.environ.get('PORT', 5001)}"
backlog = 2048

# Worker processes
workers = max(1, min(multiprocessing.cpu_count() * 2 + 1, 4))  # Cap at 4 for Render free tier
worker_class = "sync"
worker_connections = 1000
timeout = 30
keepalive = 2

# Restart workers after this many requests, to help prevent memory leaks
max_requests = 1000
max_requests_jitter = 50

# Logging
loglevel = os.environ.get('LOG_LEVEL', 'info')
errorlog = '-'  # stderr
accesslog = '-'  # stdout
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# Process naming
proc_name = 'phishguard-ai'

# Server mechanics
daemon = False
pidfile = '/tmp/gunicorn.pid'
tmp_upload_dir = None
user = None
group = None

# SSL (handled by Render)
# keyfile = None
# certfile = None

# Application
preload_app = True  # Load application code before forking workers
enable_stdio_inheritance = True

# Worker tuning
worker_tmp_dir = "/dev/shm" if os.path.exists("/dev/shm") else None

def when_ready(server):
    server.log.info("PhishGuard AI server is ready. Listening on: %s", server.address)

def worker_int(worker):
    worker.log.info("worker received INT or QUIT signal")

def pre_fork(server, worker):
    server.log.info("Worker spawned (pid: %s)", worker.pid)

def post_fork(server, worker):
    server.log.info("Worker spawned (pid: %s)", worker.pid)