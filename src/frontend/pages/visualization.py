import re
from taipy.gui import notify
import taipy as tp

from visualization import visualization_algorithms
from utils import encode_image

from frontend.pages.current_configuration import current_configuration
from frontend.pages.detailed_description import create_detailed_description, create_short_description
from frontend.pages.parameter_list import create_variable_parameter_list
from frontend.job_configs import scenario_configs


class VisualisationMember:
    def __init__(self):
        """ Variables accessible in the state """
        self.all = visualization_algorithms
        self.current = self.all[0]

        self.result_name = "Enter result file name!"
        self.selected_result = ""
        self.selected_result_url = None
        self.available_results = {"Name": []}

        self.show_image = None

    @staticmethod
    def reload(state):
        """ Reloads the variables in the pages """
        state.vis.available_results = {"Name": [
            x for x in state.vis.current.list_results()["Name"] if
            x.startswith(state.data.current.__name__)
        ]}
        state.vis = state.vis

    @staticmethod
    def update_params(state, var_name, value):
        var_name = re.search("'(.+?)'", var_name).group(1)
        try:
            dtype = type(state.vis.current.parameter[var_name])
            conversion = dtype(value)
            state.vis.current.parameter[var_name] = conversion
        except ValueError:
            notify(
                state, 'error',
                f'Parameter {var_name}: needs to be of type '
                f'{type(state.vis.current.parameter[var_name])} '
                f'but is {type(value)}'
            )
        VisualisationMember.reload(state)

    @staticmethod
    def select_visualisation(state, var_name, value):
        """ Reacts on clicks on the clustering selection """
        state.vis.current = value
        VisualisationMember.reload(state)

    @staticmethod
    def infer_model(state, id, payload):
        # Check if name is correct
        name = state.vis.result_name
        name = name.replace(" ", "_")
        if name == "Enter_result_name!" or name == "":
            notify(state, 'error', f'Please enter an result name!')
            return
        if not name.startswith(state.data.current.__name__):
            name = state.data.current.__name__ + "_" + name
        if not name.endswith(".pkl"):
            name += ".pkl"
        state.vis.current.list_results()
        if name in state.vis.current.results["Name"] or \
                name + ".pkl" in state.vis.current.results["Name"]:
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
                   f'dataset {state.dataset_selected_name}!')
            return

        scenario = tp.create_scenario(
            scenario_configs.visualize,
            name=name,
        )
        scenario.dataset.write(data)
        scenario.representation.write(representation)
        scenario.representation_result_name.write(embedding)
        scenario.visualisation.write(state.vis.current)
        scenario.visualisation_result_name.write(name)
        notify(state, 'info', f'Submitted visualisation job "{name}"')
        job = tp.submit(scenario, wait=True)
        result = scenario.finished.read()
        if result:
            notify(state, 'success', f'Visualised "{name}"')
        else:
            notify(state, 'error', f'Could not visualise "{name}"')
        state.vis.current.list_results()
        VisualisationMember.reload(state)

    @staticmethod
    def select_result(state, var_name, payload):
        """ Reacts on clicks on the result selection """
        result = state.vis.available_results["Name"][payload["index"]]
        state.vis.selected_result = result
        state.vis.selected_result_url = state.vis.current.get_result_url(result)
        VisualisationMember.reload(state)

    @staticmethod
    def refresh_result(state, id, payload):
        state.vis.current.list_results()
        VisualisationMember.reload(state)

    @staticmethod
    def delete_result(state, id, payload):
        name = state.vis.selected_result
        if name == "":
            notify(state, 'error', f'Please select a result first!')
            return
        if not state.vis.current.exist_result(name):
            notify(state, 'error', f'Result {name} does not exist!')
            return
        ret = state.vis.current.delete_result(name)
        if ret:
            notify(state, 'info', f'Deleted result {name}')
        else:
            notify(state, 'error', f'Cannot delete result {name}')
        state.vis.current.list_results()
        VisualisationMember.reload(state)

    @staticmethod
    def render_result(state, id, payload):
        result_name = state.vis.selected_result
        vis_method = state.vis.current

        if result_name == "":
            notify(state, 'error', f'Please select an result first!')
            return
        if not vis_method.exist_result(result_name):
            notify(state, 'error', f'Result {result_name} does not exist!')
            return

        cluster_result_name = state.clu.selected_result
        cluster_method = state.clu.current
        if not cluster_method.exist_result(cluster_result_name):
            labels = None
        else:
            cluster_result = cluster_method.load_result(cluster_result_name)
            labels = list(cluster_result)

        if state.data.current.data is not None:
            highlight_samples = state.data.current.data["select"]
            highlight_samples = [i for i, x in enumerate(highlight_samples) if x]
            if len(highlight_samples) == 0:
                highlight_samples = None
        else:
            highlight_samples = None
        img = vis_method.render(
            result_name,
            highlight_samples=highlight_samples,
            labels=labels
        )
        state.vis.show_image = encode_image(img)
        notify(state, 'success', f'{result_name}, {vis_method.__name__}')
        VisualisationMember.reload(state)

    @staticmethod
    def download_fail(state, id, payload):
        return


vis = VisualisationMember()

''' Create page design '''

parameter_list = create_variable_parameter_list(
    vis, "vis", "vis.update_params"
)
detailed_description = create_detailed_description(vis, "vis")
short_description = create_short_description(vis, "vis")

page = """
<|layout|columns=1 4|

"""

# Left Column
page += """
<|part|render={True}|
<|cage|
## Methods
<|{vis.current}|selector|lov={vis.all}|adapter={lambda x: x.__name__}|on_change={vis.select_visualisation}|class_name=fullwidth|>  
|>
""" + current_configuration + """
|>

"""

# Right Column "Short Description"
page += """
<|part|render={True}|
<|cage|
## <|{vis.current.__name__}|text|raw|>
""" + short_description + """
|>
"""
# Right Column - "Inference"
page += """
<|cage|
### Visualize Representations
Visualizations help to gather insights about data inherent structures. 
Modern visualization algorithms usually needs to perform a preprocessing
step to reduce the data dimensionality.
To create a new visualization, please select parameter, enter a model name and click on "VISUALIZE".
After the visualization preprocessing has been performed, it will appear in the list of available visualizations.
Select it and click on "SHOW RESULT". 
Additionally, you can click on "REFRESH" to refresh the list of available visualizations, delete a visualization by clicking on "DELETE" or download the model by clicking on the download button.

<|layout|columns=1 1|

<|part|render={True}|
#### Parameter
<|cage|
""" + parameter_list + """ 
|>
<|layout|columns=1 1|
<|part|render={True}|
<|{vis.result_name}|input|>
|>
<|part|render={True}|
<|Visualize|button|on_action={vis.infer_model}|>
|>
|>
|>

<|part|render={True}|
#### Available Results
<|{vis.available_results}|table|on_action={vis.select_result}|rebuild|>
Selected result: <|{vis.selected_result}|text|raw|>

<|layout|columns=1 1 1|
<|part|render={True}|
<|Refresh|button|on_action={vis.refresh_result}|>
|>
<|part|render={True}|
<|Delete|button|on_action={vis.delete_result}|>
|>
<|part|render={True}|
<|{vis.selected_result_url}|file_download|on_action={vis.download_fail}|name={vis.selected_result}|>
|>
|>

|>
|>
|>
"""
# Right Column - "Visualization"
page += """
<|cage|
### Data Visualization
Please select a result and click on "SHOW RESULT". Your data will be visualized!
Some tricks: 

 - You can select samples in the data view to highlight them in the visualization. just go to the data view and click on the samples you want to highlight.
 - If you have selected clustering results, they will be used to color the visualization.
 - You can reduce the data density by modifying the **visualization_fraction** parameter of the visualization algorithm. 

<|layout|columns=1 1|
<|part|render={True}|
<|Show Result|button|on_action={vis.render_result}|>
|>
<|part|render={True}|
<|...|button|on_action={vis.render_result}|>
|>
|>
<|{vis.show_image}|image|id=visualization_image|>
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
    "vis",
]
