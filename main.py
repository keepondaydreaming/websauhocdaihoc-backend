from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from model import Excerpt

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
    score = len(req.excerpt)
    return score


if __name__ == "__main__":
    pass
