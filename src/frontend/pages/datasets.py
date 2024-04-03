from taipy.gui import Gui, notify

import numpy as np

from data import datasets

from utils import encode_image
from frontend.pages.current_configuration import current_configuration


class DataMember:
    def __init__(self):
        """ Variables accessible in the state """
        self.all = datasets
        self.current = self.all[0]
        self.selected_samples = int(np.sum(self.current.data["select"]))
        self.selected_idx = 0
        self.selected_image = encode_image(
            self.current.get_image(self.selected_idx)
        )

    @staticmethod
    def reload(state):
        """ Reloads the variables in the pages """
        state.data = state.data

    @staticmethod
    def select_sample(state, var_name, payload: dict):
        """ Reacts on clicks on the sample table """
        if payload["col"] == "select":
            select = state.data.current.data["select"][payload["index"]]
            state.data.current.data.loc[payload["index"], "select"] = ~select
            state.data.selected_samples = \
                int(np.sum(state.data.current.data["select"]))

        state.data.selected_idx = payload["index"]
        state.data.selected_image = encode_image(
            state.data.current.get_image(payload["index"])
        )

        DataMember.reload(state)

    @staticmethod
    def select_dataset(state, var_name, value):
        """ Reacts on clicks on the dataset selection """
        notify(state, 'info', f'Selected dataset is: {value.__name__}')
        state.data.current = value
        state.data.selected_samples = int(
            np.sum(state.data.current.data["select"]))
        DataMember.reload(state)


data = DataMember()

__all__ = [
    "data",
]

page = """
<|layout|columns=1 4|

<|part|render={True}|
<|cage|
## Datasets
<|{data.current}|selector|lov={data.all}|adapter={lambda x: x.__name__}|on_change={data.select_dataset}|class_name=fullwidth|>  
|>

""" + current_configuration + """
|> 


<|part|render={True}|

<|cage|
## <|{data.current.__name__}|text|raw|>
<|part|render={not data.current.downloaded}|
<div id="error_alert">
Data is not downloaded! Please read the docs and download the data first.
</div> 
|>

<|{data.current.thumbnail}|image|class_name=centered|>

### Dataset Properties

<|{data.current.properties}|table|show_all=True|>
|>

<|cage|
### Data
<|{data.current.data}|table|on_action={data.select_sample}|rebuild|>
|>

|>
|>
"""
