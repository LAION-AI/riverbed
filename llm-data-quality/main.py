if __name__ == "__main__":
    import argparse
    import os
    import json
    import yaml
    from datasets import Dataset
    from llmdq import Config, llmdq_pipeline

    parser = argparse.ArgumentParser("Data quality pipeline")
    parser.add_argument("config")
    parser.add_argument("in_data_path")
    parser.add_argument("out_data_path")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    config = Config(**config)

    with open(args.in_data_path) as f:
        data = json.load(f)

    data = {"instruct": [d['instruct'] for d in data],
            "answer": [d['answer'] for d in data]}
    data = Dataset.from_dict(data)

    clustered_data, removed_dataset = llmdq_pipeline(data, config)

    base_name = os.path.splitext(args.in_data_path)[0]
    clustered_data.save_to_disk(base_name + "_filtered")
    removed_dataset.save_to_disk(base_name + "_removed")
