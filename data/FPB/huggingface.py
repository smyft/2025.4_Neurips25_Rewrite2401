from datasets import load_dataset

# Login using e.g. `huggingface-cli login` to access this dataset
ds = load_dataset("TheFinAI/en-fpb")
# print(ds.type)
print(ds)

print(ds["train"])

print(ds["train"][0])
# print(ds["train"]['id'])
