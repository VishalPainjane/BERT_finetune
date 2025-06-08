from huggingface_hub import login
from datasets import load_dataset
import pandas as pd
import os
import dotenv

dotenv.load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

# Paste your Hugging Face token here
login(HF_TOKEN)

squad = load_dataset("squad", split="train[:5000]")
squad = squad.train_test_split(test_size=0.2)

data_dict = squad["train"].to_dict()
df = pd.DataFrame.from_dict(data_dict)

df.to_csv('data/squad_data.csv', index=False)