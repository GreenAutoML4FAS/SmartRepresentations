from taipy import Gui, Core, Config
from taipy.gui import Html, Markdown

from frontend.pages import *
from frontend.theme import theme
from utils import GREENAUTOML4FAS_LOGO

DEVELOPMENT = True

style_kit = {
    "color_background_light": "#a4dbb3",
    "color_background_dark": "#a4dbb3",
    "color_paper_light": "#67c281",
    "color_paper_dark": "#67c281",
    "color_primary": "#3f9e5a",
    "color_secondary": "#a25221",
}

Config.configure_job_executions(
    # mode="standalone", nb_of_workers=1,
    mode="development",
)
Core().run()

Gui(
    pages=pages,
    css_file="style.css"
).run(
    title="GreenAutoML4FAS",
    favicon=GREENAUTOML4FAS_LOGO,
    use_reloader=DEVELOPMENT,
    stylekit=style_kit
)
