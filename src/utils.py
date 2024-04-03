import os
import cv2
import io


def encode_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    is_success, buffer = cv2.imencode(".png", img)
    io_buf = io.BytesIO(buffer)
    return io_buf.getvalue()


ROOT_DIR = os.path.join(__file__.split("src")[0], "data")
GREENAUTOML4FAS_LOGO = os.path.join("assets", "images", "AutoML4FAS_Logo.jpeg")
GOV_LOGO = os.path.join("assets", "images", "Bund.png")
VISCODA_LOGO = os.path.join("assets", "images", "Viscoda.png")
LUH_LOGO = os.path.join("assets", "images", "LUH.png")

ASSET_PATH = os.path.join("assets")
