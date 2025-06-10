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

# train data
data_dict = squad["train"].to_dict()
df = pd.DataFrame.from_dict(data_dict)

df.to_csv('data/squad_data_train.csv', index=False)

# test data
data_dict_test = squad["test"].to_dict()
df_test = pd.DataFrame.from_dict(data_dict_test)

df.to_csv('data/squad_data_test.csv', index=False)