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
        return JSONResponse(status_code=500, content={"detail": "API Key is missing in Vercel environment variables."})
    
    try:
        genai.configure(api_key=api_key)
        # Using the exact recommended model string directly
        model = genai.GenerativeModel('gemini-3.8-flash')
        
        system_context = "You are A3 QC AI, an advanced AI assistant built by Team 48, Punjab College Quaid Campus (ICS CQ-31). Keep answers clean, professional, and structured."
        full_prompt = f"{system_context}\n\nModule: {module}\nQuery: {prompt_text}"
        
        response = model.generate_content(full_prompt)
        return {"result": response.text}
    except Exception as e:
        # Fallback instantly to standard gemini-1.5-flash if 3.8 encounters any issue
        try:
            model = genai.GenerativeModel('gemini-1.5-flash')
            response = model.generate_content(full_prompt)
            return {"result": response.text}
        except Exception as inner_e:
            return JSONResponse(status_code=500, content={"detail": f"AI Error: {str(inner_e)}"})

@app.post("/api/classify-waste")
async def classify_waste(
    description: str = Form(default=""),
    file: UploadFile = File(default=None)
):
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return JSONResponse(status_code=500, content={"detail": "API Key is missing in Vercel environment variables."})
    
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-3.8-flash')
        
        contents = ["Classify this waste into Plastic, Glass, Metal, Organic, E-waste, or Paper, with recycling steps."]
        if file and file.filename != "":
            image_bytes = await file.read()
            image = Image.open(io.BytesIO(image_bytes))
            contents.append(image)
        elif description:
            contents.append(f"Description: {description}")
        
        response = model.generate_content(contents)
        return {"result": response.text}
    except Exception as e:
        try:
            model = genai.GenerativeModel('gemini-1.5-flash')
            response = model.generate_content(contents)
            return {"result": response.text}
        except Exception as inner_e:
            return JSONResponse(status_code=500, content={"detail": f"AI Error: {str(inner_e)}"})
