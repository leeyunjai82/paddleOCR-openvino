import openvino as ov
from pathlib import Path
MODEL_NAME = "paddle_multi"
MODEL_DIR = Path("model")

model_xml = Path(MODEL_NAME).with_suffix(".xml")
if not model_xml.exists():
    ov_model = ov.convert_model("Multilingual_PP-OCRv3_det_infer/inference.pdmodel")
    ov.save_model(ov_model, str(model_xml))
else:
    print(f"{model_xml} already exists.")

