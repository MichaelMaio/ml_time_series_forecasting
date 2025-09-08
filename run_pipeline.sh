#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Build all containers.
docker-compose build

# Run pipeline sequentially.
# Remove the default network after each step because otherwise we get errors.
docker network rm ml_time_series_forecasting_default >/dev/null 2>&1
docker-compose run trainer 2>/dev/null
docker network rm ml_time_series_forecasting_default >/dev/null 2>&1
docker-compose run promoter 2>/dev/null
docker network rm ml_time_series_forecasting_default >/dev/null 2>&1
docker-compose run predictor 2>/dev/null
