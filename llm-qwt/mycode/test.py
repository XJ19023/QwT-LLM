from datasets import get_dataset_config_names, get_dataset_split_names

# 查看可用的 config 名
print(get_dataset_config_names("allenai/c4"))

# 查看某个 config（比如 en）下的 split 名
print(get_dataset_split_names("allenai/c4", config_name="en"))
