import splitfolders

splitfolders.ratio(
    "Breeds_dataset",   # your extracted folder
    output="dataset",   # new folder will be created
    seed=42,
    ratio=(0.8, 0.2)    # 80% train, 20% validation
)
