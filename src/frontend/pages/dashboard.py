import os
from utils import GREENAUTOML4FAS_LOGO, GOV_LOGO, VISCODA_LOGO, LUH_LOGO

dashboard_logo = GREENAUTOML4FAS_LOGO
dashboard_credit_logo = GOV_LOGO
dashboard_vis_logo = VISCODA_LOGO
dashboard_luh_logo = LUH_LOGO

page = """
<|layout|columns=1 1 1|

<|part|render={True}|
<|cage|
### About the Project 
GreenAutoML4FAS is a lighthouse project of a consortium consisting of
TNT, IMS, AI, and Viscoda. The goal is to provide a tool that allows to easily 
analyse datastructures with novel clustering and representation algorithms.
The project is funded by the German Federal Ministry of the Environment, Nature 
Conservation, Nuclear Safety and Consumer Protection.
|>
  
<|cage|
### Running Jobs 
<|None|job_selector|don't show_id|don't show_submitted_id|don't show_submission_id|don't show_submitted_label|>
|>
|>

<|part|render={True}|
<|cage|
### Datasets
<||selector|lov={data.all}|adapter={lambda x: x.__name__}|id=my_selector|class_name=fullwidth|>   

### Representations 
<||selector|lov={rep.all}|adapter={lambda x: x.__name__}|class_name=fullwidth|>  

### Clustering Algorithms
<||selector|lov={clu.all}|adapter={lambda x: x.__name__}|class_name=fullwidth|>  

### Visualization Algorithms 
<||selector|lov={vis.all}|adapter={lambda x: x.__name__}|class_name=fullwidth|>  

|>
|>

<|part|render={True}|

<|cage|
### Credits

This work was partially supported by the German Federal Ministry of the Environment, Nature Conservation, Nuclear Safety and Consumer Protection (GreenAutoML4FAS project no. 67KI32007A).

<|{dashboard_credit_logo}|image|class_name=centered|>

Partners:

<|{dashboard_vis_logo}|image|class_name=centered|>


<|{dashboard_luh_logo}|image|class_name=centered|>

|>





|>
|>

"""

__all__ = [
    "dashboard_logo",
    "dashboard_credit_logo",
    "dashboard_vis_logo",
    "dashboard_luh_logo",
]
