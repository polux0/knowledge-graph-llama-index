import chromadb
from utils.environment_setup import load_environment_variables

env_vars = load_environment_variables()
# Initialize the Chroma client
# Ensure that you replace 'your_api_key' with your actual Chroma API key
remote_db = chromadb.HttpClient(
    host=env_vars["CHROMA_URL"], port=env_vars["CHROMA_PORT"]
)

# Fetch all collections
collections = remote_db.list_collections()

# Loop through each collection and delete it
for collection in collections:
    try:
        remote_db.delete_collection(name=str(collection.name))
        print(f"Deleted collection: {str(collection.name)}")
    except Exception as e:
        print(f"Failed to delete collection: {str(collection.id)}, Error: {str(e)}")
