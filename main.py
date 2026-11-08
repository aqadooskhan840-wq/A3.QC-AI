from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from google import genai
import os
from PIL import Image
import io

app = FastAPI(title="A3 QC AI Platform API")

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def read_index():
    from fastapi.responses import FileResponse
    return FileResponse("static/index.html")

@app.post("/api/generate")
async def generate_ai(
    module: str = Form(...),
    prompt_text: str = Form(...)
):
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return JSONResponse(status_code=500, content={"detail": "API Key is missing in Vercel environment variables."})
    
    try:
        client = genai.Client(api_key=api_key)
        
        system_context = "You are A3 QC AI, an advanced AI assistant built by Team 48, Punjab College Quaid Campus (ICS CQ-31). Keep answers clean, professional, and structured."
        full_prompt = f"{system_context}\n\nModule: {module}\nQuery: {prompt_text}"
        
        # Using the standard stable gemini-1.5-flash model
        response = client.models.generate_content(
            model='gemini-1.5-flash',
            contents=full_prompt,
        )
        return {"result": response.text}
    except Exception as e:
        return JSONResponse(status_code=500, content={"detail": f"AI Error: {str(e)}"})

@app.post("/api/classify-waste")
async def classify_waste(
    description: str = Form(default=""),
    file: UploadFile = File(default=None)
):
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return JSONResponse(status_code=500, content={"detail": "API Key is missing in Vercel environment variables."})
    
    try:
        client = genai.Client(api_key=api_key)
        contents = ["Classify this waste into Plastic, Glass, Metal, Organic, E-waste, or Paper, with recycling steps."]
        
        if file and file.filename != "":
            image_bytes = await file.read()
            image = Image.open(io.BytesIO(image_bytes))
            contents.append(image)
        elif description:
            contents.append(f"Description: {description}")
        
        response = client.models.generate_content(
            model='gemini-1.5-flash',
            contents=contents,
        )
        return {"result": response.text}
    except Exception as e:
        return JSONResponse(status_code=500, content={"detail": f"AI Error: {str(e)}"})
