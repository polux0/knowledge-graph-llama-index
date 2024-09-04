# RAG Setup Guide

## Overview

This repository provides a complete setup for deploying a knowledge graph indexing system, including:

1. Elasticsearch schema setup and migrations.
2. Redis user setup and ACL configuration.
3. Automatic Elasticsearch snapshots using cron jobs.
4. Docker-based infrastructure to run embedding processes and a Streamlit UI for interaction.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Decomposed Scripts](#decomposed-scripts)
    - [Create Directories and Set Permissions](#create-directories-and-set-permissions)
    - [Setup Elasticsearch Schema and Migrations](#setup-elasticsearch-schema-and-migrations)
    - [Setup Elasticsearch Cron Job](#setup-elasticsearch-cron-job)
    - [Setup Redis ACL](#setup-redis-acl)
3. [Running the Setup](#running-the-setup)
4. [Docker Configuration](#docker-configuration)
5. [Running the Docker Containers](#running-the-docker-containers)
6. [Environment Files Setup](#environment-files-setup)

---

## Prerequisites

Before starting, ensure you have the following installed:

- [Docker](https://docs.docker.com/get-docker/) & [Docker Compose](https://docs.docker.com/compose/install/)
- SSH access to the remote DigitalOcean Droplet (or similar environment).
- Access to GitHub secrets for environment variables like `REDIS_USERNAME`, `REDIS_PASSWORD`, etc.

---

## Decomposed Scripts

### 1. **Create Directories and Set Permissions**

Script: [`/scripts/core/setup_directories_and_permissions.sh`](scripts/core/setup_directories_and_permissions.sh)

This script creates the necessary directories for Elasticsearch, Redis, Neo4j, Chroma, and Nebula, and sets proper permissions.

**Commands Executed:**
- Creates required directories for Elasticsearch, Redis, Neo4j, etc.
- Sets `chmod 777` permissions for critical directories.
- Changes ownership to user `1000:1000` where needed.
- Executes the following scripts:
  - `/scripts/redis/setup_redis_acl.sh`
  - `/scripts/elasticsearch/setup_elasticsearch.sh`
  - `/scripts/elasticsearch/setup_elasticsearch_cron.sh`

### 2. **Setup Elasticsearch Schema and Migrations**

Script: [`/scripts/elasticsearch/setup_elasticsearch.sh`](scripts/elasticsearch/setup_elasticsearch.sh)

This script checks and runs the Elasticsearch schema setup and applies the required migrations.

### 3. **Setup Elasticsearch Cron Job**

Script: [`/scripts/elasticsearch/setup_elasticsearch_cron.sh`](scripts/elasticsearch/setup_elasticsearch_cron.sh)Ru

This script installs a cron job that takes snapshots of Elasticsearch data every hour.

### 4. **Setup Redis ACL**

Script: [`/scripts/redis/setup_redis_acl.sh`](scripts/redis/setup_redis_acl.sh)

This script configures Redis users and ACL by creating the file `/usr/local/etc/redis/users.acl` and allowing you to enter usernames and passwords during setup.

---

## Running the Setup

### Step 1: Execute Setup Scripts

1. Run the directories and permissions setup script:
   ```bash
   ./scripts/core/setup_directories_and_permissions.sh
   

Next, manually run the Redis ACL setup:
    ```./scripts/redis/setup_redis_acl.sh```
    

#### Important: 

Save the ```usernames``` and ```passwords``` you entered. You'll need to set them in ```.env.production``` and ```.env.api.production``` files under ```REDIS_USERNAME``` and ```REDIS_PASSWORD```.


### Step 2: Configure Environment Files

Create files ```.env.production``` and ```.env.api.production```. 

Use the ```env.production.copy``` and ```.env.api.production.copy``` files as guidance to structure your environment configuration.


### Docker Configuration

### Step 1: Run the MRI Indexing

Open ```docker-compose.yaml``` and navigate to the ```app``` service/section. 
You'll need to modify it depending on which indexing process or UI you want to run.

### Step 2: Run the MRI Indexing

Comment out all other commands in the `app` section/service of `docker-compose.yaml` except:

```command: python ./modules/multi_representation_indexing.py```

Build and run the Docker containers:

```docker-compose up --build```

Wait until the embeddings process finishes. You should see the following log:

```Created MRI embeddings for complete documentation...```

Bring down the Docker containers:

```docker-compose down```

### Step 3: Run the RAPTOR Indexing

Comment out all other commands in the ```app``` service/section of ```docker-compose.yaml``` except:
    
```command: python ./modules/create_raptor_indexing_langchain.py```

Build and run the Docker containers again:

```docker-compose up --build```

Wait for the process to complete. The console will log:

```Created RAPTOR embeddings for complete documentation.```

Bring down the Docker containers:

```docker-compose down```

### Step 4: Enable the Streamlit UI

Modify ```docker-compose.yaml``` one last time by commenting out all other commands except:
    
```command: streamlit run ./modules/app.py```

Build and run the Docker containers to start the Streamlit UI:

```docker-compose up --build```

You now have the Streamlit UI running and ready for interaction.

### Environment Files Setup

Ensure that your ```.env.production``` and ```.env.api.production``` files are properly configured based on the ```.copy``` files provided.

Set your ```REDIS_USERNAME``` and ```REDIS_PASSWORD``` to the values you used during the Redis ACL setup.