import re
from taipy.gui import notify
import taipy as tp

from representations import representations

from frontend.pages.current_configuration import current_configuration
from frontend.pages.parameter_list import create_variable_parameter_list
from frontend.pages.detailed_description import create_detailed_description, create_short_description
from frontend.job_configs import scenario_configs


class RepresentationsMember:
    def __init__(self):
        """ Variables accessible in the state """
        self.all = representations
        self.current = self.all[0]

        self.model_name = "Enter model name!"
        self.selected_model = ""
        self.selected_model_url = None

        self.result_name = "Enter embedding name!"
        self.selected_result = ""
        self.selected_result_url = None
        self.available_results = {"Name": []}

    @staticmethod
    def reload(state):
        """ Reloads the variables in the pages """
        state.rep.available_results = {"Name": [
            x for x in state.rep.current.list_results()["Name"] if
            x.startswith(state.data.current.__name__)
        ]}
        state.rep = state.rep

    @staticmethod
    def update_params(state, var_name, value):
        var_name = re.search("'(.+?)'", var_name).group(1)
        try:
            dtype = type(state.rep.current.parameter[var_name])
            conversion = dtype(value)
            state.rep.current.parameter[var_name] = conversion
        except ValueError:
            notify(
                state, 'error',
                f'Parameter {var_name}: needs to be of type '
                f'{type(state.rep.current.parameter[var_name])} '
                f'but is {type(value)}'
            )
        RepresentationsMember.reload(state)

    @staticmethod
    def select_representation(state, var_name, value):
        """ Reacts on clicks on the representation selection """
        state.rep.current = value
        RepresentationsMember.reload(state)

    @staticmethod
    def select_model(state, var_name, payload):
        """ Reacts on clicks on the model selection """
        model = state.rep.current.models["Name"][payload["index"]]
        state.rep.selected_model = model
        state.rep.selected_model_url = state.rep.current.get_model_url(model)
        RepresentationsMember.reload(state)

    @staticmethod
    def load_model(state, id, payload):
        name = state.rep.selected_model
        ret = state.rep.current.load_model(name)
        if ret:
            notify(state, 'info', f'Loaded model {name}')
        else:
            notify(state, 'error', f'Cannot load model {name}')
        RepresentationsMember.reload(state)

    @staticmethod
    def refresh_model(state, id, payload):
        state.rep.current.list_models()
        RepresentationsMember.reload(state)

    @staticmethod
    def delete_model(state, id, payload):
        name = state.rep.selected_model
        if name == "":
            notify(state, 'error', f'Please select a model first!')
            return
        if not state.rep.current.exist_model(name):
            notify(state, 'error', f'Model {name} does not exist!')
            return
        ret = state.rep.current.delete_model(name)
        if ret:
            notify(state, 'info', f'Deleted model {name}')
        else:
            notify(state, 'error', f'Cannot delete model {name}')
        state.rep.current.list_models()
        RepresentationsMember.reload(state)

    @staticmethod
    def train_model(state, id, payload):
        # Check if name is correct
        name = state.rep.model_name
        name = name.replace(" ", "_")
        if name == "Enter_model_name!" or name == "":
            notify(state, 'error', f'Please enter a model name!')
            return
        if name in state.rep.current.models["Name"] or \
                name + ".pkl" in state.rep.current.models["Name"]:
            notify(state, 'error', f'Model name already exists!')
            return
        # Create and submit Scenario
        scenario = tp.create_scenario(
            scenario_configs.train_representation,
            name=name,
        )
        scenario.dataset.write(state.data.current)
        scenario.representation.write(state.rep.current)
        scenario.representation_result_name.write(name)
        notify(state, 'info', f'Submitted training job for model "{name}"')
        job = tp.submit(scenario, wait=True)
        # Evaluate result
        result = scenario.finished.read()
        if result:
            notify(state, 'success', f'Trained model "{name}"')
        else:
            notify(state, 'error', f'Could not train model "{name}"')
        state.rep.current.list_models()
        RepresentationsMember.reload(state)

    @staticmethod
    def infer_model(state, id, payload):
        # Check if name is correct
        name = state.rep.result_name
        name = name.replace(" ", "_")
        if name == "Enter_embedding_name!" or name == "":
            notify(state, 'error', f'Please enter an embedding name!')
            return
        if not name.startswith(state.data.current.__name__):
            name = state.data.current.__name__ + "_" + name
        if not name.endswith(".pkl"):
            name += ".pkl"
        state.rep.current.list_results()
        if name in state.rep.current.results["Name"] or \
                name + ".pkl" in state.rep.current.results["Name"]:
            notify(state, 'error', f'Result name {name} already exists!')
            return
        # Create and submit Scenario
        scenario = tp.create_scenario(
            scenario_configs.infer_representation,
            name=name,
        )
        scenario.dataset.write(state.data.current)
        scenario.representation.write(state.rep.current)
        scenario.representation_result_name.write(name)
        notify(state, 'info', f'Submitted inference job for model "{name}"')
        job = tp.submit(scenario, wait=True)
        # Evaluate result
        result = scenario.finished.read()
        if result:
            notify(state, 'success', f'Inferred embedding "{name}"')
        else:
            notify(state, 'error', f'Could not infer embedding "{name}"')
        state.rep.current.list_results()
        RepresentationsMember.reload(state)

    @staticmethod
    def select_result(state, var_name, payload):
        """ Reacts on clicks on the result selection """
        result = state.rep.available_results["Name"][payload["index"]]
        state.rep.selected_result = result
        state.rep.selected_result_url = state.rep.current.get_result_url(result)
        RepresentationsMember.reload(state)

    @staticmethod
    def refresh_result(state, id, payload):
        state.rep.current.list_results()
        RepresentationsMember.reload(state)

    @staticmethod
    def delete_result(state, id, payload):
        name = state.rep.selected_result
        if name == "":
            notify(state, 'error', f'Please select a result first!')
            return
        if not state.rep.current.exist_result(name):
            notify(state, 'error', f'Result {name} does not exist!')
            return
        ret = state.rep.current.delete_result(name)
        if ret:
            notify(state, 'info', f'Deleted result {name}')
        else:
            notify(state, 'error', f'Cannot delete result {name}')
        state.rep.current.list_results()
        RepresentationsMember.reload(state)

    @staticmethod
    def download_fail(state, id, payload):
        return


rep = RepresentationsMember()

''' Create page design '''

parameter_list = create_variable_parameter_list(
    rep, "rep", "rep.update_params"
)
detailed_description = create_detailed_description(rep, "rep")
short_description = create_short_description(rep, "rep")

page = """
<|layout|columns=1 4|

"""

# Left Column
page += """
<|part|render={True}|
<|cage|
## Methods
<|{rep.current}|selector|lov={rep.all}|adapter={lambda x: x.__name__}|on_change={rep.select_representation}|class_name=fullwidth|>  
|>
""" + current_configuration + """
|>

"""

# Right Column "Short Description"
page += """
<|part|render={True}|
<|cage|
## <|{rep.current.__name__}|text|raw|>
""" + short_description + """
|>
"""
# Right Column - "Training"
page += """
<|cage|
### Train new Model
Representations describe the data in a way that is easier to process for the
clustering algorithms. Before a representation can be inferred, a model has to
be trained/defined.
To create a new model, please select parameter, enter a model name and click on "TRAIN".
After the model has been trained, it will appear in the list of available models.
Select the model and click on "LOAD" to load the model. 
Additionally, you can click on "REFRESH" to refresh the list of available models, delete a model by clicking on "DELETE" or download the model by clicking on the download button.

<|layout|columns=1 1|
<|part|render={True}|
#### Parameter
<|cage|
""" + parameter_list + """ 
|>
<|layout|columns=1 1|
<|part|render={True}|
<|{rep.model_name}|input|>
|>
<|part|render={True}|
<|Train|button|on_action={rep.train_model}|>
|>
|>
|>

<|part|render={True}|
#### Available Models
<|{rep.current.models}|table|on_action={rep.select_model}|rebuild|>

Selected model: <|{rep.selected_model}|text|raw|>

<|layout|columns=1 1 1 1|
<|part|render={True}|
<|Load|button|on_action={rep.load_model}|>
|>
<|part|render={True}|
<|Refresh|button|on_action={rep.refresh_model}|>
|>
<|part|render={True}|
<|Delete|button|on_action={rep.delete_model}|>
|>
<|part|render={True}|
<|{rep.selected_model_url}|file_download|on_action={rep.download_fail}|name={rep.selected_model}|>
|>
|>
|>
|>
|>

"""
# Right Column - "Inference"
page += """
<|cage|
### Create Representations 
After a model has been trained, you can create a representation for the dataset.
To infer a new representation, please enter a name for the result and click on "CREATE".
After the representation has been inferred, it will appear in the list of available representations.
Select the representation and click on "REFRESH" to refresh the list of available representations, delete a representation by clicking on "DELETE" or download the representation by clicking on the download button.

<|layout|columns=1 1|
<|part|render={True}|
#### Create Representations
<|layout|columns=1 1|
<|part|render={True}|
<|{rep.result_name}|input|>
|>
<|part|render={True}|
<|Create|button|on_action={rep.infer_model}|>
|>
|>
|>

<|part|render={True}|
#### Available Representations
<|{rep.available_results}|table|on_action={rep.select_result}|rebuild|>
Selected result: <|{rep.selected_result}|text|raw|>

<|layout|columns=1 1 1|
<|part|render={True}|
<|Refresh|button|on_action={rep.refresh_result}|>
|>
<|part|render={True}|
<|Delete|button|on_action={rep.delete_result}|>
|>
<|part|render={True}|
<|{rep.selected_result_url}|file_download|on_action={rep.download_fail}|name={rep.selected_result}|>
|>
|>

|>
|>
|>
"""
# Right Column - "Long Description"
page += """
<|cage|
## Detailed Method Description
""" + detailed_description + """
|>
|>
|>
"""

__all__ = [
    "rep",
]
