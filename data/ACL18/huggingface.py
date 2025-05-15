from datasets import load_dataset

ds = load_dataset("TheFinAI/flare-sm-acl")

print(ds["test"])

print(ds["test"][0])
