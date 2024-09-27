#!/bin/bash

check_services_up() {
  # Get a list of running services as an array
  mapfile -t services < <(docker-compose ps --services --filter "status=running")
  
  # Get the list of expected services as an array
  mapfile -t expected_services < <(docker-compose config --services)

  # Compare running services with expected services
  if [ "${#services[@]}" -eq "${#expected_services[@]}" ]; then
    echo "All services are up and running."
    return 0
  else
    echo "Some services are still starting or not running."
    return 1
  fi
}

# Wait for all services to be up
while ! check_services_up; do
  echo "Waiting for all services to start..."
  sleep 5
done

echo "All services are running and docker-compose up is finished!"
