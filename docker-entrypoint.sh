#!/bin/bash
set -e

echo "Starting entrypoint script..."

# Use SSH_KEY_NAME env var if set, default to id_rsa if not
SSH_KEY_NAME=${SSH_KEY_NAME:-id_rsa}
SSH_KEY_PATH="/root/.ssh/${SSH_KEY_NAME}"

# Set correct permissions for SSH key
if [ -f "${SSH_KEY_PATH}" ]; then
    echo "SSH key found at ${SSH_KEY_PATH}, setting permissions..."
    chmod 600 "${SSH_KEY_PATH}"
    ls -la "${SSH_KEY_PATH}"
else
    echo "No SSH key found at ${SSH_KEY_PATH}"
fi

echo "Executing command: $@"
# Execute the command passed to docker run
exec "$@"