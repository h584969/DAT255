from fastai.vision.all import *
from PIL import Image
from typing import Annotated
from fastapi import FastAPI, File, UploadFile
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

learn: Module = load_learner("syltagurk")
learn.eval()


print("spinning up api")



app = FastAPI(debug=True)


class PredictResponse(BaseModel):
    name: str
    amt: float

@app.post("/api/v1/predict")
async def predict(file: UploadFile = File(...)) -> PredictResponse:
    # Be aware that the file needs to be saved before opening with PIL.Image
    try:
        contents = await file.read()  # Read content
        image = Image.open(io.BytesIO(contents))
        image = image.resize((256,256))

        name, idx, values = learn.predict(image)
        print(name, idx, values)  
        return PredictResponse(name=name, amt=values[idx.item()])
    except BaseException as e:
        print(e)
    return PredictResponse(name="lol", amt=1.0)


app.mount("/", StaticFiles(directory="static"), name="static")

# image = Image.open("images/finger_sponge/finger_sponge001.jpg")
# image.resize((256,256))


# predictions, _ = learn.predict(image)

# image.show()
# print(predictions)
