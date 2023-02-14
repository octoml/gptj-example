import uvicorn
from fastapi import FastAPI
from gptj import GPTJModel

# FastAPI server setup
app = FastAPI()
gptj: GPTJModel = None


@app.get("/generate")
def generate(prompt: str, length: int = 128):
    result = gptj.generate(prompt, length)
    return result


def launch(port: int, model: GPTJModel):
    global gptj
    gptj = model
    config = uvicorn.Config("server:app", port=port, log_level="info")
    server = uvicorn.Server(config)
    server.run()
