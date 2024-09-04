import os
from utils.environment_setup import load_environment_variables
from langfuse import Langfuse
from multi_representation_indexing import generate_response_based_on_multirepresentation_indexing

env_vars = load_environment_variables()

os.environ["LANGFUSE_PUBLIC_KEY"] = env_vars["LANGFUSE_PUBLIC_KEY"]
os.environ["LANGFUSE_SECRET_KEY"] = env_vars["LANGFUSE_SECRET_KEY"]
os.environ["LANGFUSE_HOST"] = env_vars["NEXTAUTH_URL"]

# init
langfuse = Langfuse()

# create a dataset
#TODO: This is fixed at the moment:
dataset_name = "initial_test"
#TODO: This is a variable that should not be in code. Where should we place it?
langfuse.create_dataset(name=dataset_name);

initial_test_items = [
    {"input": {"question": "What are the factors that determine the sustainability of mineral resources, and how do they impact resource management?"}, 
    "expected_output": "The sustainability of mineral resources is determined by several key factors, including the renewability, recyclability, and energy expenditure required to reverse entropy. Mineral resources, particularly metallic minerals, can often be recycled with relatively low energy expenditure, making their use more sustainable. However, minerals like coal and hydrocarbons are considered non-renewable due to the high entropy created during their use, which makes them nearly impossible to recycle efficiently. The sustainability of mineral resources impacts resource management by necessitating careful planning to ensure that minerals are used efficiently, recycled where possible, and that alternatives are identified in case of shortages. Sustainable mineral resource management involves balancing current human needs with the preservation of resources for future generations."},
    {"input": {"question": "How does the unified habitat heritage network approach resource planning, and what is its significance in community-type societies?"}, 
    "expected_output": "The unified habitat heritage network approach to resource planning treats a habitat as a single technical unit, composed of various sub-habitat technical units. This approach focuses on identifying, acquiring, and optimizing the use of resources within a global network of habitats. It integrates resource availability with the needs of human fulfillment, ensuring that all resources are managed in a way that supports the long-term sustainability of the habitat. The significance of this approach in community-type societies lies in its ability to systematically plan and allocate resources to meet human needs while minimizing environmental impact. It promotes the efficient use of resources, the recycling of materials, and the development of sustainable technologies, making it a cornerstone of sustainable community planning."},
]

# upload to langfuse

for item in initial_test_items:
  langfuse.create_dataset_item(
      dataset_name=dataset_name, #TODO: This is a variable that should not be in code. Where should we place it?
      # any python object or value
      input=item["input"],
      # any python object or value, optional
      expected_output=item["expected_output"]
)

# Simple evaluation definition
def simple_evaluation(output, expected_output):
  return output == expected_output

# Run the experiment
def run_langchain_experiment(experiment_name: str):
  dataset = langfuse.get_dataset(dataset_name)
 
  for item in dataset.items:
    handler = item.get_langchain_handler(run_name=experiment_name)
 
    completion = generate_response_based_on_multirepresentation_indexing(item.input["question"], 1, handler)
 
    handler.trace.score(
      name="exact_match",
      value=simple_evaluation(completion, item.expected_output)
    )

run_langchain_experiment("Initial test1")