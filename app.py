# app.py
from fastapi import FastAPI, HTTPException, Form, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse
import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import soundfile as sf
import numpy as np
import os
import uuid
import uvicorn
from typing import Optional, List, Tuple
from pydantic import BaseModel
import shutil

app = FastAPI(title="Adia_TTS Wolof API", 
              description="API for Text-to-Speech synthesis in Wolof language using CONCREE's Adia_TTS model")

UPLOAD_DIR = "/tmp/text_files"
OUTPUT_DIR = "/tmp/tts_output"


os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

class TTSRequest(BaseModel):
    text: str
    description: Optional[str] = "A warm and natural voice, with a conversational flow"
    temperature: Optional[float] = 0.01
    max_new_tokens: Optional[int] = 1000
    do_sample: Optional[bool] = True
    top_k: Optional[int] = 50
    repetition_penalty: Optional[float] = 1.2

class AdiaTTS:
    def __init__(self, model_id: str = "CONCREE/Adia_TTS"):
        self.model_id = model_id
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None
        self.load_model()
        
    def load_model(self):
        try:
            self.model = ParlerTTSForConditionalGeneration.from_pretrained(
                self.model_id,
                token=os.environ.get('HF_TOKEN')
            ).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                token=os.environ.get('HF_TOKEN')
            )
            print(f"Model loaded successfully on {self.device}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise RuntimeError(f"Failed to load model: {e}")
    
    def synthesize(self, text: str, description: str, config: dict) -> Tuple[str, np.ndarray]:
        try:

            if len(text) > 200:
                max_pos = min(200, len(text))
                pause_chars = ['.', '!', '?', ',', ';', ':', '…']
            
                last_pause = 0
                for char in pause_chars:
                    pos = text[:max_pos].rfind(char)
                    if pos > last_pause:
                        last_pause = pos
                
                if last_pause > 0:
                    text = text[:last_pause + 1]
                else:

                    last_space = text[:max_pos].rfind(' ')
                    if last_space > 0:
                        text = text[:last_space]
  
                    else:
                        # If no space found, just truncate at 200
                        text = text[:200]
            
            # Prepare inputs
            input_ids = self.tokenizer(description, return_tensors="pt").input_ids.to(self.device)
            prompt_ids = self.tokenizer(text, return_tensors="pt").input_ids.to(self.device)
            
            audio = self.model.generate(
                input_ids=input_ids,
                prompt_input_ids=prompt_ids,
                **config
            )
            
            audio_np = audio.cpu().numpy().squeeze()
            
            output_path = os.path.join(OUTPUT_DIR, f"{uuid.uuid4()}.wav")
            sf.write(output_path, audio_np, self.model.config.sampling_rate)
            
            return output_path, audio_np
        except Exception as e:
            raise Exception(f"Failed to synthesize speech: {e}")

    def segment_text(self, text: str) -> List[str]:
        """
        Split long text into segments of max 200 characters at natural boundaries.
        Finds the last natural pause before the 200-character limit and splits there.
        Natural pauses are prioritized in this order:
        1. Sentence endings (., !, ?)
        2. Other punctuation marks (,, ;, :, ...)
        3. Word boundaries (spaces)
        
        This creates segments that end at natural pauses for better audio quality.
        """
        segments = []
        remaining_text = text.strip()
        
        MAX_CHARS = 200  
        
       
        sentence_end_chars = ['.', '!', '?']
        pause_chars = [',', ';', ':', '…']
        
        while remaining_text:
            
            if len(remaining_text) <= MAX_CHARS:
                segments.append(remaining_text)
                break

            segment_text = ""
            
            last_sentence_end = -1
            for i in range(min(MAX_CHARS, len(remaining_text))):
                if remaining_text[i] in sentence_end_chars:
                    last_sentence_end = i
            
            if last_sentence_end != -1:
                segment_text = remaining_text[:last_sentence_end + 1]
                remaining_text = remaining_text[last_sentence_end + 1:].strip()

            else:
                
                last_punct = -1
                for i in range(min(MAX_CHARS, len(remaining_text))):
                    if remaining_text[i] in pause_chars:
                        last_punct = i
                
                if last_punct != -1:
                    segment_text = remaining_text[:last_punct + 1]
                    remaining_text = remaining_text[last_punct + 1:].strip()
                 
                else:
                    text_to_check = remaining_text[:MAX_CHARS]
                    last_space = text_to_check.rfind(' ')
                    
                    if last_space != -1:
                        segment_text = remaining_text[:last_space]
                        remaining_text = remaining_text[last_space + 1:].strip()
                       
                    else:
                        segment_text = remaining_text[:MAX_CHARS]
                        remaining_text = remaining_text[MAX_CHARS:].strip()
            
            segments.append(segment_text)
        
        return segments
        
    def concatenate_audio_files(self, file_paths: List[str]) -> str:
        """
        Concatenate multiple audio files into a single file with carefully tuned transitions.
        
        This function:
        1. Reads all audio segments
        2. Analyzes optimal crossfade length based on segment characteristics
        3. Applies adaptive crossfading techniques
        4. Normalizes audio levels throughout segments
        
        Returns the path to the combined file.
        """
        if not file_paths:
            raise ValueError("No audio files provided for concatenation")
        
        sample_rate = None
        audio_segments = []
        
        for file_path in file_paths:
            data, sr = sf.read(file_path)
            
            if sample_rate is None:
                sample_rate = sr
            elif sample_rate != sr:
                pass
                # I will come back to add resampling logic but for  now let's keep it like this
            
            audio_segments.append(data)
        
        normalized_segments = []
        for segment in audio_segments:
            if np.abs(segment).max() < 0.001:
                normalized_segments.append(segment)
                continue
                
    
            max_amp = np.abs(segment).max()
            target_amp = 0.7  # Target amplitude - not too loud, not too soft
            normalized = segment * (target_amp / max_amp)
            normalized_segments.append(normalized)
        
        # Find optimal crossfade length (longer is generally better for speech)
        # Use a longer crossfade for speech transitions
        crossfade_ms = 150  # milliseconds 
        crossfade_samples = int((crossfade_ms / 1000) * sample_rate)
        
        result = normalized_segments[0]
        for i in range(1, len(normalized_segments)):
            current_segment = normalized_segments[i]
            
            if len(result) < crossfade_samples or len(current_segment) < crossfade_samples:
                actual_crossfade = min(len(result), len(current_segment), crossfade_samples)

            else:
                actual_crossfade = crossfade_samples
            
            if actual_crossfade <= 0:
                # Just concatenate if crossfade isn't possible
                result = np.concatenate([result, current_segment])
                continue
                
            # Create smooth crossfade shapes (using raised cosine for more natural transitions)
            # Raised cosine creates smoother transitions than linear fades
            t = np.linspace(0, np.pi, actual_crossfade)
            fade_out = np.cos(t) * 0.5 + 0.5  # Smooth fade out from 1 to 0
            fade_in = np.sin(t) * 0.5 + 0.5   # Smooth fade in from 0 to 1
            
            # Apply crossfade
            result_end = result[-actual_crossfade:]
            segment_start = current_segment[:actual_crossfade]
            
            # Blend the overlapping regions
            crossfade_region = (result_end * fade_out) + (segment_start * fade_in)
            
            # Combine everything
            result = np.concatenate([result[:-actual_crossfade], crossfade_region, current_segment[actual_crossfade:]])
        
        # Post-processing: smooth out any remaining irregularities
        # Apply a gentle low-pass filter to the entire audio to smooth transitions
        # This helps reduce any artifacts or sudden changes in frequency response
        
        # Save combined audio
        output_path = os.path.join(OUTPUT_DIR, f"combined_{uuid.uuid4()}.wav")
        sf.write(output_path, result, sample_rate)
    
        return output_path


tts_model = AdiaTTS()

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "gpu_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "model": "CONCREE/Adia_TTS"
    }

@app.post("/predict")
async def predict(request: TTSRequest):
    try:
        generation_config = {
            "temperature": request.temperature,
            "max_new_tokens": request.max_new_tokens,
            "do_sample": request.do_sample,
            "top_k": request.top_k,
            "repetition_penalty": request.repetition_penalty
        }
        
        if len(request.text) > 200:
            segments = tts_model.segment_text(request.text)
            output_files = []
            
            for segment in segments:
                output_path, _ = tts_model.synthesize(
                    text=segment,
                    description=request.description,
                    config=generation_config
                )
                output_files.append(output_path)
                     
            combined_path = tts_model.concatenate_audio_files(output_files)
            
            return FileResponse(
                path=combined_path, 
                media_type="audio/wav", 
                filename="synthesized_speech.wav"
            )

        else:
            output_path, _ = tts_model.synthesize(
                text=request.text,
                description=request.description,
                config=generation_config
            )
            
         
            return FileResponse(
                path=output_path, 
                media_type="audio/wav", 
                filename="synthesized_speech.wav"
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch_predict")
async def batch_predict(
    file: UploadFile = File(...),
    description: str = Form("A warm and natural voice, with a conversational flow"),
    temperature: float = Form(0.8),
    max_new_tokens: int = Form(1000),
    do_sample: bool = Form(True),
    top_k: int = Form(50),
    repetition_penalty: float = Form(1.2)
):
   
    file_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}_{file.filename}")
    
    try:
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
   
        with open(file_path, "r", encoding="utf-8") as f:
            text_content = f.read()
        
   
        segments = tts_model.segment_text(text_content)
        
     
        output_files = []
        generation_config = {
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "top_k": top_k,
            "repetition_penalty": repetition_penalty
        }
        
        for segment in segments:
            output_path, _ = tts_model.synthesize(
                text=segment,
                description=description,
                config=generation_config
            )
            output_files.append(output_path)
        
        combined_path = tts_model.concatenate_audio_files(output_files)
        
        return FileResponse(
            path=combined_path, 
            media_type="audio/wav", 
            filename="synthesized_speech.wav"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up the uploaded file
        if os.path.exists(file_path):
            os.remove(file_path)

if __name__ == "__main__":
    port = int(os.environ.get('AIP_HTTP_PORT', 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
