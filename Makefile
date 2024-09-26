run:
	@echo "Running host-side setup scripts..."
	@./scripts/core/setup_directories_and_permissions.sh
	@./scripts/redis/setup_redis_acl.sh
	@echo "Starting docker-compose..."
	@docker-compose up -d
	@echo "Waiting for containers to be healthy..."
	@sleep 15
	@echo "Running Elasticsearch setup scripts..."
	@./scripts/elasticsearch/setup_elasticsearch.sh
	@./scripts/elasticsearch/setup_repository.sh

stop:
	@echo "Stopping docker-compose..."
	@docker-compose down
