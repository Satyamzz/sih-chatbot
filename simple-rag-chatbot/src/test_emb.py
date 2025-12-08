import os
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
load_dotenv()
client = InferenceClient(
    provider="hf-inference",
    api_key=os.environ["HF_TOKEN"],
)

result = client.sentence_similarity(
    "The weather is lovely today",             
    other_sentences=[                          
        "It is raining heavily outside",
        "The sun is shining effectively",
        "I love machine learning"
    ]
)
print(result)