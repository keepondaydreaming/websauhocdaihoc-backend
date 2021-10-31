from pydantic import BaseModel


class Excerpt(BaseModel):
    excerpt: str
