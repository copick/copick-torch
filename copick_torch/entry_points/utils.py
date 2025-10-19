import yaml


# Create a custom dumper that uses flow style for lists only.
class InlineListDumper(yaml.SafeDumper):
    def represent_list(self, data):
        node = super().represent_list(data)
        node.flow_style = True  # Use inline style for lists
        return node


def save_parameters_yaml(params: dict, output_path: str):
    """
    Save parameters to a YAML file.
    """
    InlineListDumper.add_representer(list, InlineListDumper.represent_list)
    with open(output_path, "w") as f:
        yaml.dump(
            params,
            f,
            Dumper=InlineListDumper,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
            encoding="utf-8",
        )
