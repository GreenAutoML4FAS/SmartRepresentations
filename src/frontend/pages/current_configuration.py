current_configuration = """
<|cage|
## Configuration
|Setting|Value|   
|-------|-----|
|Dataset:|<|{data.current.__name__}|text|raw|>|
|Representation:|<|{rep.current.__name__}|text|raw|>|
|Embedding:|<|{rep.selected_result}|text|raw|>|
|Cluster Method:|<|{clu.current.__name__}|text|raw|>|
|Cluster Result:|<|{clu.selected_result}|text|raw|>|
|Visualization Method:|<|{vis.current.__name__}|text|raw|>|
|Visualization Result:|<|{vis.selected_result}|text|raw|>|
|>

<|cage|
|Setting|Value|   
|-------|-----|
|Selected Samples:|<|{data.selected_samples}|text|raw|>|
|Shown Image sample:| <|{data.selected_idx}|text|raw|>|

<|{data.selected_image}|image|id=dataset_sample|>
|>

"""

__all__ = ["current_configuration"]
