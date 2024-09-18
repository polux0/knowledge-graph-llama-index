# RAG Setup Guide

## Overview

This repository provides a complete setup for deploying a retrieval augmented generation system, including:

1. Elasticsearch schema setup and migrations.
2. Redis user setup and ACL configuration.
3. Automatic Elasticsearch snapshots using cron jobs.
4. Docker-based infrastructure to run embedding processes as well as Streamlit UI and Telegram for interaction.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Decomposed Scripts](#decomposed-scripts)
    - [Create Directories and Set Permissions](#create-directories-and-set-permissions)
    - [Setup Elasticsearch Schema and Migrations](#setup-elasticsearch-schema-and-migrations)
    - [Setup Elasticsearch Cron Job](#setup-elasticsearch-cron-job)
    - [Setup Redis ACL](#setup-redis-acl)
3. [Running the Setup](#running-the-setup)
    - [Deploy locally](#deploy-locally)
    - [Deploy remotely](#deploy-remotely)
4. [Docker Configuration](#docker-configuration)
5. [Running the Docker Containers](#running-the-docker-containers)
6. [Environment Files Setup](#environment-files-setup)

---

## Prerequisites

Before starting, ensure you have the following installed:

- [Docker](https://docs.docker.com/get-docker/) & [Docker Compose](https://docs.docker.com/compose/install/)
- SSH access to the remote server ( DigitalOcean Droplet or similar environment ).
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

Script: [`/scripts/elasticsearch/setup_elasticsearch_cron.sh`](scripts/elasticsearch/setup_elasticsearch_cron.sh)

This script installs a cron job that takes snapshots of Elasticsearch data every hour.

### 4. **Setup Redis ACL**

Script: [`/scripts/redis/setup_redis_acl.sh`](scripts/redis/setup_redis_acl.sh)

This script configures Redis users and ACL by creating the file `/usr/local/etc/redis/users.acl` and allowing you to enter usernames and passwords during setup.

---

## Running the Setup

### Deploy locally:

### Step 1: Execute Setup Scripts

1. Run the directories and permissions setup script:
   ```bash
   . /scripts/core/setup_directories_and_permissions.sh
   

2. Next, manually run the Redis ACL setup:
    ```. scripts/redis/setup_redis_acl.sh```

3. Next, manually run ```. scripts/elasticsearch/setup_elasticsearch.sh```


4. Finally, manually run ```. scripts/elasticsearch/setup_elasticsearch_cron.sh```


#### Important: 

Save the ```usernames``` and ```passwords``` you entered. You'll need to set them in ```.env.production```, ```.env.api.production``` and ```.env.ui.production```files under ```REDIS_USERNAME``` and ```REDIS_PASSWORD```.


### Step 2: Configure Environment Files

Create files ```.env.production```, ```.env.api.production``` and ```env.ui.production```. 

Use the ```env.production.copy```, ```.env.api.production.copy``` and ```.env.api.production.copy``` files as guidance to structure your environment configuration.

#### Important: 

Embedding the whole documentation might last between 6 - 12h per index. 

In case you would like to give it relatively quick try, modify:

```documents_directory = "../data/documentation_optimal/"``` 

to 

```documents_directory = "../data/documentation_optimal/test"``` in ```multi_representation_indexing.py``` 

as well as 

```folders = ['decision-system', 'habitat-system', 'lifestyle-system', 'material-system', 'project-execution', 'project-plan','social-system', 'system-overview']```

to 

```folders = ['test1']``` in ```create_raptor_indexing_langchain.py```

In case you decide to create the indexes via embeddings with different data set, you'll need to manually change name of the collection: 

```chroma_collection_name = "MRITESTTTTTTTTTTT4"```
```redis_namespace = "parent-documents-MRITESTTTTTTTTTTT4"```

in ```multi_representation_indexing.py``` 

and 

```chroma_collection_name = "raptor-locll-test12"```

in ```create_raptor_indexing_langchain.py```

### Docker Configuration

### Step 1: Run the MRI Indexing

Open ```docker-compose.yaml``` and navigate to the ```app``` service/section. 
You'll need to modify it depending on which indexing process or UI you want to run.

### Step 2: Run the MRI Indexing


Comment out those services:

```langfuse```, ```postgres```, ```api```, ```telegram_bot```

as they are not relevant in this stage of the process in `docker-compose.yaml`. 

Comment out all other commands in the `app` section/service of `docker-compose.yaml` except:

```command: python ./modules/multi_representation_indexing.py```

Build and run the Docker containers:

```docker-compose up --build```

Wait until the embeddings process finishes. You should see the following log:

```Created MRI embeddings for complete documentation...```

Bring down the Docker containers:

```docker-compose down```

### Step 3: Run the RAPTOR Indexing

Comment out those services if you have not already:

```langfuse```, ```postgres```, ```api```, ```telegram_bot```

as they are not relevant in this stage of the process in `docker-compose.yaml`. 

Comment out all other commands in the ```app``` service/section of ```docker-compose.yaml``` except:
    
```command: python ./modules/create_raptor_indexing_langchain.py```

Build and run the Docker containers again:

```docker-compose up --build```

Wait for the process to complete. The console will log:

```Created RAPTOR embeddings for complete documentation.```

Bring down the Docker containers:

```docker-compose down```

### Step 4: Create telegram bot

[Create telegram bot](https://core.telegram.org/bots/tutorial) and save token in both ```.env.production``` and ```.env.api.production``` under ```TELEGRAM_DEVELOPMENT_INTEGRATION_TOKEN```. This will enable you to interact with the bot via telegram

### Step 5: Comment out the app service in ```docker-compose.yaml``` in order to save on resources

There are issues with streamlit that needs to be fixed, so this is only temporary. 

If you succesfully commented out ```app``` service, it should look like this:

![Logo](https://i.ibb.co/YR4frk0/app.png)

### Environment Files Setup

Ensure that your ```.env.production```, ```.env.api.production``` and ```.env.api.production``` files are properly configured based on the ```.copy``` files provided.

Set your ```REDIS_USERNAME``` and ```REDIS_PASSWORD``` to the values you used during the Redis ACL setup.

Uncomment out services that you previously commented out:

```langfuse```, ```postgres```, ```api```, ```telegram_bot```

Finally run `docker compose up --build`

### Deploy remotely ( via Github actions ):

### 1. **Fork the repository**

1. **Go to the Repository**:  
   Navigate to the repository you want to fork on GitHub.

2. **Fork the Repository**:  
   In the top-right corner, click `Fork`. GitHub will create a copy of the repository under your account.

3. **Set Up GitHub Secrets in the Fork**:  
   After forking, go to your forked repository and follow the steps above to create your own secrets for deployment.

By forking the repository and setting up your own secrets, you can customize and deploy the solution to your remote servers.

### 2. **Creating GitHub Secrets**

1. **Navigate to Your Repository**:  
   Open the repository page on GitHub.

2. **Open Settings**:  
   Click the `Settings` tab at the top of the repository.

3. **Access Secrets**:  
   In the sidebar under `Security`, click `Secrets and variables` > `Actions`.

4. **Add a New Secret**:  
   Click `New repository secret`.

5. **Name the Secret**:  
   Enter a name using uppercase letters and underscores, e.g., `API_KEY`, `DB_PASSWORD`.

6. **Add the Secret Value**:  
   Paste the sensitive value (e.g., API key, token) in the `Secret` field.

7. **Save the Secret**:  
   Click `Add secret` to save it.
   
### List of Necessary Secrets to Add
- `DROPLET_IP_ADDRESS=XXXXXXXXXXXXXXXXXXXX` (The public IP address of the remote server, used to establish a connection for deployment and management.)
- `SSH_PRIVATE_KEY=XXXXXXXXXXXXXXXXXXXX` (This is the private key component of an SSH key pair. It must match the public key that has been added to the remote server's authorized keys, allowing secure authentication and access to the server via SSH.)
- `HUGGING_FACE_INFERENCE_ENDPOINT=XXXXXXXXXXXXXXXXXXXX`
- `HUGGING_FACE_API_KEY=XXXXXXXXXXXXXXXXXXXX`
- `NEO4J_USERNAME=neo4j`
- `NEO4J_PASSWORD=XXXXXXXXXXXXXXXXXXXX`
- `NEO4J_URL=bolt://neo4j:7687`
- `NEO4J_DATABASE=neo4j`
- `CHROMA_URL=chromadb`
- `CHROMA_PORT=8000`
- `ELASTIC_SCHEME=http`
- `ELASTIC_URL=elasticsearch`
- `ELASTIC_PORT=9200`
- `OPENAI_API_KEY=XXXXXXXXXXXXXXXXXXXX`
- `NEBULA_URL=graphd`
- `NEBULA_PORT=9669`
- `NEBULA_USERNAME=XXXXXXXXXXXXXXXXXXXX`
- `NEBULA_PASSWORD=XXXXXXXXXXXXXXXXXXXX`
- `REDIS_HOST=redis`
- `REDIS_PORT=6379`
- `REDIS_USERNAME1=XXXXXXXXXXXXXXXXXXXX`
- `REDIS_PASSWORD1=XXXXXXXXXXXXXXXXXXXX`
- `REDIS_USERNAME2=XXXXXXXXXXXXXXXXXXXX`
- `REDIS_PASSWORD2=XXXXXXXXXXXXXXXXXXXX`
- `COHERE_API_KEY=XXXXXXXXXXXXXXXXXXXX`
- `ENV=production`
- `TELEGRAM_DEVELOPMENT_INTEGRATION_TOKEN=XXXXXXXXXXXXXXXXXXXX`
- `API_URL=http://api:5000`
- `GROQ_API_KEY=XXXXXXXXXXXXXXXXXXXX`
- `DATABASE_URL=XXXXXXXXXXXXXXXXXXXX`
- `LANGFUSE_PORT=3000`
- `NEXTAUTH_SECRET=XXXXXXXXXXXXXXXXXXXX`
- `NEXTAUTH_URL=XXXXXXXXXXXXXXXXXXXX`
- `SALT=XXXXXXXXXXXXXXXXXXXX`
- `ENCRYPTION_KEY=XXXXXXXXXXXXXXXXXXXX`
- `POSTGRES_DB=XXXXXXXXXXXXXXXXXXXX`
- `POSTGRES_USER=XXXXXXXXXXXXXXXXXXXX`
- `POSTGRES_PASSWORD=XXXXXXXXXXXXXXXXXXXX`

By forking the repository and setting up these secrets, you can customize and deploy the solution to your remote servers.