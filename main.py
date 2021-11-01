from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from model import Excerpt
from roberta import Inference

model = Inference(model_path='model_1.pth')

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def read_root():
    return {"Hello": "World"}


@app.post("/inference")
async def inference(req: Excerpt):
    if not req.excerpt or req.excerpt.isspace():
        return 0.0

    score = model.predict(req.excerpt.strip())
    return score


if __name__ == "__main__":
    pass
