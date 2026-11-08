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
        raise HTTPException(status_code=500, detail="Server Gemini API Key is not configured in environment variables.")
    
    try:
        genai.configure(api_key=api_key)
        # Correct stable model name
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        system_context = "You are A3 QC AI, an advanced multi-module AI assistant built by Team 48, Punjab College Quaid Campus (ICS CQ-31 Section) under Prof. Naveed Asif. Provide clean, professional, and well-structured answers."
        full_prompt = f"{system_context}\n\nTask/Module: {module}\nUser Input: {prompt_text}"
        
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
        raise HTTPException(status_code=500, detail="Server Gemini API Key is not configured in environment variables.")
    
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        contents = ["Classify this waste item into Plastic, Glass, Metal, Organic, E-waste, or Paper, and provide eco-friendly recycling steps."]
        
        if file and file.filename != "":
            image_bytes = await file.read()
            image = Image.open(io.BytesIO(image_bytes))
            contents.append(image)
        elif description:
            contents.append(f"Waste Item Description: {description}")
        else:
            return JSONResponse({"result": "Please provide an image or text description of the waste item."})
            
        response = model.generate_content(contents)
        return {"result": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
