from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from app import main

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

class TopicRequest(BaseModel):
    topic: str

@app.get("/")                          
async def root():                      
    return FileResponse("static/index.html")
@app.post("/generate")
async def generate(request: TopicRequest):
    result = await main(request.topic)
    return {"article": result}
