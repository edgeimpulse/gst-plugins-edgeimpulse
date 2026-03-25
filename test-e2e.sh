#!/bin/bash
set -e

# Run e2e tests in Docker.
#
# Prerequisites:
#   export EI_PROJECT_ID="your-project-id"
#   export EI_API_KEY="your-api-key"
#
# Or point to a local model:
#   export EI_MODEL=~/Downloads/your-model-directory
#
# Usage:
#   ./test-e2e.sh              # run all tests
#   ./test-e2e.sh test_name    # run a specific test

if [ -z "$EI_PROJECT_ID" ] && [ -z "$EI_MODEL" ]; then
    echo "Error: Set EI_PROJECT_ID + EI_API_KEY, or EI_MODEL to run e2e tests."
    echo ""
    echo "  export EI_PROJECT_ID=your-project-id"
    echo "  export EI_API_KEY=your-api-key"
    echo "  ./test-e2e.sh"
    echo ""
    echo "  # Or with a local model:"
    echo "  export EI_MODEL=~/Downloads/your-model-directory"
    echo "  ./test-e2e.sh"
    exit 1
fi

TEST_FILTER=""
if [ -n "$1" ]; then
    TEST_FILTER="-- $1 --nocapture"
    echo "Running test: $1"
else
    TEST_FILTER="-- --nocapture"
    echo "Running all e2e tests"
fi

docker compose -f docker-compose.test.yml build e2e-test

# Override CMD to pass test filter
docker compose -f docker-compose.test.yml run --rm e2e-test \
    cargo test --release --test e2e $TEST_FILTER
