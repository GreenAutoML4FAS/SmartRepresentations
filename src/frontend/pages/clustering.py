import re
from taipy.gui import notify
import taipy as tp

from clustering import cluster_algorithms

from frontend.pages.current_configuration import current_configuration
from frontend.pages.detailed_description import create_detailed_description, create_short_description
from frontend.pages.parameter_list import create_variable_parameter_list
from frontend.job_configs import scenario_configs


class ClusteringMember:
    def __init__(self):
        """ Variables accessible in the state """
        self.all = cluster_algorithms
        self.current = self.all[0]

        self.result_name = "Enter embedding name!"
        self.selected_result = ""
        self.selected_result_url = None
        self.available_results = {"Name": []}

    @staticmethod
    def reload(state):
        """ Reloads the variables in the pages """
        state.clu.available_results = {"Name": [
            x for x in state.clu.current.list_results()["Name"] if
            x.startswith(state.data.current.__name__)
        ]}
        state.clu = state.clu

    @staticmethod
    def update_params(state, var_name, value):
        var_name = re.search("'(.+?)'", var_name).group(1)
        try:
            dtype = type(state.clu.current.parameter[var_name])
            conversion = dtype(value)
            state.clu.current.parameter[var_name] = conversion
        except ValueError:
            notify(
                state, 'error',
                f'Parameter {var_name}: needs to be of type '
                f'{type(state.clu.current.parameter[var_name])} '
                f'but is {type(value)}'
            )
        ClusteringMember.reload(state)

    @staticmethod
    def select_clustering(state, var_name, value):
        """ Reacts on clicks on the clustering selection """
        state.clu.current = value
        ClusteringMember.reload(state)

    @staticmethod
    def download_fail(state, id, payload):
        return

    @staticmethod
    def infer_model(state, id, payload):
        # Check if name is correct
        name = state.clu.result_name
        name = name.replace(" ", "_")
        if name == "Enter_result_name!" or name == "":
            notify(state, 'error', f'Please enter an result name!')
            return
        if not name.startswith(state.data.current.__name__):
            name = state.data.current.__name__ + "_" + name
        if not name.endswith(".pkl"):
            name += ".pkl"
        state.clu.current.list_results()
        if name in state.clu.current.results["Name"] or \
                name + ".pkl" in state.clu.current.results["Name"]:
            notify(state, 'error', f'Result name {name} already exists!')
            return
        # Create and submit Scenario
        data = state.data.current
        representation = state.rep.current
        embedding = state.rep.selected_result
        if not representation.exist_result(embedding):
            notify(state, 'error',
                   f'Representation result {embedding} does not exist!')
            return
        if not embedding.startswith(state.data.current.__name__):
            notify(state, 'error',
                   f'Representation result {embedding} does not match '
                   f'dataset {state.data.current}!')
            return

        scenario = tp.create_scenario(
            scenario_configs.infer_clustering,
            name=name,
        )
        scenario.dataset.write(data)
        scenario.representation.write(representation)
        scenario.representation_result_name.write(embedding)
        scenario.clustering.write(state.clu.current)
        scenario.clustering_result_name.write(name)
        notify(state, 'info', f'Submitted Cluster job "{name}"')
        job = tp.submit(scenario, wait=True)
        result = scenario.finished.read()
        if result:
            notify(state, 'success', f'Clustered "{name}"')
        else:
            notify(state, 'error', f'Could not cluster "{name}"')
        state.clu.current.list_results()
        ClusteringMember.reload(state)

    @staticmethod
    def select_result(state, var_name, payload):
        """ Reacts on clicks on the result selection """
        result = state.clu.available_results["Name"][payload["index"]]
        print(result)
        state.clu.selected_result = result
        state.clu.selected_result_url = state.clu.current.get_result_url(result)
        ClusteringMember.reload(state)

    @staticmethod
    def refresh_result(state, id, payload):
        state.clu.current.list_results()
        ClusteringMember.reload(state)

    @staticmethod
    def delete_result(state, id, payload):
        name = state.clu.selected_result
        if name == "":
            notify(state, 'error', f'Please select a result first!')
            return
        if not state.clu.current.exist_result(name):
            notify(state, 'error', f'Result {name} does not exist!')
            return
        ret = state.clu.current.delete_result(name)
        if ret:
            notify(state, 'info', f'Deleted result {name}')
        else:
            notify(state, 'error', f'Cannot delete result {name}')
        state.clu.current.list_results()
        ClusteringMember.reload(state)


clu = ClusteringMember()

''' Create page design '''

parameter_list = create_variable_parameter_list(
    clu, "clu", "clu.update_params"
)
detailed_description = create_detailed_description(clu, "clu")
short_description = create_short_description(clu, "clu")

page = """
<|layout|columns=1 4|

"""

# Left Column
page += """
<|part|render={True}|
<|cage|
## Methods
<|{clu.current}|selector|lov={clu.all}|adapter={lambda x: x.__name__}|on_change={clu.select_clustering}|class_name=fullwidth|>  
|>
""" + current_configuration + """
|>

"""

# Right Column "Short Description"
page += """
<|part|render={True}|
<|cage|
## <|{clu.current.__name__}|text|raw|>

""" + short_description + """

|>
"""
# Right Column - "Inference"
page += """
<|cage|
### Cluster Representations 
You can cluster the representations of the dataset with the selected clustering
algorithm. The result will be saved in a file and can be used for
visualization in next steps.
If you want to cluster the representations, please enter a name for the resulting
embedding file and press the "Cluster" button. The result will be saved in the results list
below.
Additionally, you can download the result file by clicking on the download
button or delete the result by clicking on the delete button.
<|layout|columns=1 1|
<|part|render={True}|
#### Parameter
<|cage|
""" + parameter_list + """ 
|>
<|layout|columns=1 1|
<|part|render={True}|
<|{clu.result_name}|input|>
|>
<|part|render={True}|
<|Cluster|button|on_action={clu.infer_model}|>
|>
|>
|>

<|part|render={True}|
#### Available Cluster Results
<|{clu.available_results}|table|on_action={clu.select_result}|rebuild|>
Selected result: <|{clu.selected_result}|text|raw|>

<|layout|columns=1 1 1|
<|part|render={True}|
<|Refresh|button|on_action={clu.refresh_result}|>
|>
<|part|render={True}|
<|Delete|button|on_action={clu.delete_result}|>
|>
<|part|render={True}|
<|{clu.selected_result_url}|file_download|on_action={clu.download_fail}|name={clu.selected_result}|>
|>
|>

|>
|>
|>
"""
# Right Column - "Long Description"
page += """
<|cage|
### Detailed Description
""" + detailed_description + """
|>
|>
|>
"""

__all__ = [
    "clu",
]
