#!/bin/sh
set -e
API_URL="${DOCUMIND_API_URL:-http://localhost:8002}"
printf 'window.__DOCUMIND__ = { apiUrl: "%s" };\n' "${API_URL%/}" > /usr/share/nginx/html/env.js
