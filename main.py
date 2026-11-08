from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import google.generativeai as genai
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
        raise HTTPException(status_code=500, detail="API Key is missing in Vercel settings.")
    
    try:
        genai.configure(api_key=api_key)
        # Using the standard stable model supported by the updated SDK
        model = genai.GenerativeModel('gemini-2.0-flash')
        
        system_context = "You are A3 QC AI, an advanced AI assistant built by Team 48, Punjab College Quaid Campus (ICS CQ-31). Keep answers clean and professional."
        full_prompt = f"{system_context}\n\nModule: {module}\nQuery: {prompt_text}"
        
        response = model.generate_content(full_prompt)
        return {"result": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/classify-waste")
async def classify_waste(
    description: str = Form(default=""),
    file: UploadFile = File(default=None)
):
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="API Key is missing in Vercel settings.")
    
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.0-flash')
        
        contents = ["Classify this waste into Plastic, Glass, Metal, Organic, E-waste, or Paper, with recycling steps."]
        if description:
            contents.append(f"Description: {description}")
        
        response = model.generate_content(contents)
        return {"result": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
