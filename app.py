import gradio as gr
import requests
import io
import os
import logging
import assemblyai as aai
from PIL import Image, ImageDraw, ImageFont
from moviepy.editor import ImageSequenceClip, AudioFileClip, CompositeVideoClip, ImageClip
from moviepy.config import change_settings
from moviepy.editor import *
import edge_tts
import asyncio
import tempfile
import json
from typing import Dict, List, Tuple
import time
import numpy as np
#import whisperx
import shutil # Added for directory operations

# Initialize logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

change_settings({"IMAGEMAGICK_BINARY": None})  # Disable ImageMagick

# Keys and other configurations
ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY")
CLOUDFLARE_API_TOKEN = os.getenv("CLOUDFLARE_API_TOKEN")
CLOUDFLARE_ACCOUNT_ID = os.getenv("CLOUDFLARE_ACCOUNT_ID")
CLOUDFLARE_MODEL = "@cf/stabilityai/stable-diffusion-xl-base-1.0"
MODELSLAB_API_KEY = os.getenv("MODELSLAB_API_KEY")

# Initialize AssemblyAI
aai.settings.api_key = ASSEMBLYAI_API_KEY

# Available voices with display names
AVAILABLE_VOICES = {
    "Male (American)": "en-US-GuyNeural - Male (American)",
    "Female (American)": "en-US-JennyNeural - Female (American)",
    "Male (British)": "en-GB-RyanNeural - Male (British)",
    "Female (British)": "en-GB-SoniaNeural - Female (British)",
    "Male (Australian)": "en-AU-WilliamNeural - Male (Australian)",
    "Female (Australian)": "en-AU-NatashaNeural - Female (Australian)",
    "Male (Hindi)": "hi-IN-MadhurNeural - Male (Hindi)",
    "Female (Hindi)": "hi-IN-SwaraNeural - Female (Hindi)"
}

def clear_temp_folder():
    """Clears the contents of the 'temp' folder."""
    temp_dir = "temp"
    if os.path.exists(temp_dir):
        logger.info(f"Clearing temporary folder: {temp_dir}")
        shutil.rmtree(temp_dir) # Removes the directory and all its contents
    os.makedirs(temp_dir, exist_ok=True) # Recreate the directory
    logger.info(f"Temporary folder {temp_dir} cleared and recreated.")



async def text_to_speech(text: str, voice: str = "en-US-GuyNeural - Male (American)") -> str:
    """
    Convert text to speech using edge-tts.
    
    Args:
        text (str): Text to convert to speech
        voice (str): Voice to use for TTS with display name
    
    Returns:
        str: Path to the generated audio file
    """
    try:
        # Extract voice ID from the display string
        voice_id = voice.split(" - ")[0] if " - " in voice else voice
        
        # Create output directory if it doesn't exist
        os.makedirs("temp", exist_ok=True)
        
        # Generate unique filename
        timestamp = int(time.time())
        output_file = f"temp/speech_{timestamp}.mp3"
        
        logger.info(f"Generating TTS with voice: {voice_id}")
        
        # Configure voice settings
        communicate = edge_tts.Communicate(text, voice_id, volume='+0%')
        
        # Convert to audio
        await communicate.save(output_file)
        
        if os.path.exists(output_file):
            logger.info(f"Successfully generated audio file: {output_file}")
            return output_file
        else:
            logger.error("Failed to generate audio file")
            raise gr.Error("Failed to generate audio file")
            
    except Exception as e:
        logger.error(f"Error in text_to_speech: {e}")
        raise gr.Error(f"Error in text-to-speech generation: {str(e)}")

'''
def generate_subtitles_whisper(audio_path: str) -> List[Tuple[float, float, str]]:
    """
    Generate word-level subtitles using WhisperX (for Hindi or Hinglish).
    
    Args:
        audio_path (str): Path to audio file (.mp3)
    Returns:
        List of (start_time, end_time, word) tuples
    """
    logger.info("Loading WhisperX model...")
    model = whisperx.load_model("base", device="cpu", compute_type="int8")
    logger.info(f"Transcribing {audio_path} using WhisperX")
    result = model.transcribe(audio_path, language="hi")  # force Hindi mode
    if "segments" not in result or not result["segments"]:
        logger.warning("No segments returned from WhisperX")
        return []
    logger.info("Aligning word-level timestamps...")
    model_a, metadata = whisperx.load_align_model(language_code="hi", device="cpu")
    result_aligned = whisperx.align(result["segments"], model_a, metadata, audio_path, device="cpu")
    words = result_aligned.get("word_segments", [])
    subtitles = [(w["start"], w["end"], w["text"]) for w in words if w["text"].strip()]
    
    logger.info(f"Generated {len(subtitles)} subtitle word-timings from WhisperX.")
    return subtitles
'''

def create_text_image(current_word: str, context_words: List[str], size=(1920, 200)) -> np.ndarray:
    """Create a text image using PIL with word highlighting."""
    # Create a new image with an RGBA mode for transparency
    image = Image.new('RGBA', size, (0, 0, 0, 0))  # Fully transparent background
    draw = ImageDraw.Draw(image)
    
    # Try to use Arial or a system font with larger size
    try:
        font = ImageFont.truetype("arial.ttf", 60)  # Increased font size
    except:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 60)
        except:
            font = ImageFont.load_default()
    
    # Join all words with spaces
    full_text = " ".join(context_words)
    
    # Get text bounding box
    bbox = draw.textbbox((0, 0), full_text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    # Calculate position to center the text
    x = (size[0] - text_width) // 2
    y = (size[1] - text_height) // 2

    # Draw each word with appropriate color
    current_x = x
    for word in context_words:
        word_with_space = word + " "
        word_bbox = draw.textbbox((0, 0), word_with_space, font=font)
        word_width = word_bbox[2] - word_bbox[0]
        
        # Highlight current word in yellow, others in white
        color = (255, 255, 0, 255) if word == current_word else (255, 255, 255, 255)
        draw.text((current_x, y), word_with_space, font=font, fill=color)
        current_x += word_width

    return np.array(image)

def get_context_words(words: List[Dict], current_idx: int, context_size: int = 5) -> List[str]:
    """Get surrounding words for context."""
    start_idx = max(0, current_idx - context_size)
    end_idx = min(len(words), current_idx + context_size + 1)
    return [word.text for word in words[start_idx:end_idx]]

def generate_subtitles(audio_path: str) -> List[ImageClip]:
    """Generate subtitle clips from audio using AssemblyAI."""
    try:
        if not audio_path or not os.path.exists(audio_path):
            logger.error(f"Invalid audio path: {audio_path}")
            return None

        # Initialize transcription
        if not aai.settings.api_key:
            logger.error("AssemblyAI API key not found")
            return None

        transcriber = aai.Transcriber()
        logger.info(f"Uploading audio file: {audio_path}")
        with open(audio_path, "rb") as audio_file:
            transcript = transcriber.transcribe(audio_file)

        words = transcript.words
        if not words:
            logger.error("No words found in transcript")
            return None

        logger.info(f"Received {len(words)} words from transcription")

        subtitles = []
        for idx, word in enumerate(words):
            try:
                # Get context words
                context_words = get_context_words(words, idx)
                
                # Create image with text
                img_array = create_text_image(word.text, context_words)
                
                # Convert to ImageClip
                clip = ImageClip(img_array, transparent=True)  # Enable transparency
                
                # Set timing
                start_time = float(word.start) / 1000
                duration = float(word.end - word.start) / 1000
                
                # Position and time the clip
                clip = (clip
                       .set_position(('center', 'center'))
                       .set_start(start_time)
                       .set_duration(duration))
                
                subtitles.append(clip)
                logger.info(f"Successfully created clip for word: {word.text}")
                
            except Exception as clip_error:
                logger.error(f"Failed to create clip for word '{word.text}'. Error: {clip_error}")
                continue

        if not subtitles:
            logger.error("No valid subtitle clips were created")
            return None

        logger.info(f"Successfully generated {len(subtitles)} subtitle clips")
        return subtitles

    except Exception as e:
        logger.error(f"Error generating subtitles: {e}")
        logger.exception("Full traceback:")
        return None

def generate_images(prompts: List[str], api_choice: str = "Cloudflare") -> Tuple[List[Image.Image], List[str]]:
    """
    Generate images using the selected API (Cloudflare or ModelsLab) from a list of text prompts.
    
    Args:
        prompts (List[str]): List of prompts
        api_choice (str): The selected API ("Cloudflare" or "ModelsLab")
    Returns:
        Tuple[List[Image.Image], List[str]]: Generated images and their file paths
    """
    if not prompts or not any(p.strip() for p in prompts):
        raise gr.Error("Prompt list is empty.")
    
    try:
        images = []
        image_filenames = []
        
        common_negative_prompt = ("blurry, low quality, low resolution, grainy, overexposed, "
                                  "underexposed, poorly drawn, mutated hands, deformed face, ugly, "
                                  "watermark, signature, text, bad anatomy, extra limbs, "
                                  "missing fingers, long neck, distorted, cropped, out of frame, "
                                  "jpeg artifacts, low contrast, duplicated features, bad proportions, "
                                  "wrong perspective, glitch, nsfw, sexual, nude, naked, disfigured, "
                                  "photorealistic, 3d render, anime, horror, scary, unsettling, "
                                  "emotionless, lifeless eyes, stiff posture, unnatural expression, "
                                  "creepy smile, zombie, plastic texture, cinematic lighting, logo, "
                                  "background clutter, extra fingers, fewer fingers, malformed limbs, disfigured arms, disfigured legs, fused fingers, extra digits")

        for idx, prompt in enumerate(prompts, start=1):
            logger.info(f"Generating image {idx}/{len(prompts)}: {prompt}")
            image_data = None

            if api_choice == "Cloudflare":
                logger.info("Using Cloudflare API for image generation.")
                if not CLOUDFLARE_API_TOKEN or not CLOUDFLARE_ACCOUNT_ID:
                    raise gr.Error("Missing Cloudflare credentials.")
                
                cf_url = f"https://api.cloudflare.com/client/v4/accounts/{CLOUDFLARE_ACCOUNT_ID}/ai/run/{CLOUDFLARE_MODEL}"
                cf_headers = {
                    "Authorization": f"Bearer {CLOUDFLARE_API_TOKEN}",
                    "Content-Type": "application/json"
                }
                cf_payload = {
                    "prompt": prompt, 
                    "negative_prompt": common_negative_prompt, 
                    "width": 1080, 
                    "height": 1920
                }
                response = requests.post(cf_url, headers=cf_headers, json=cf_payload)
                if response.status_code != 200:
                    logger.error(f"Cloudflare Error {response.status_code}: {response.text}")
                    raise gr.Error(f"Cloudflare Error {response.status_code}: {response.text}")
                image_data = response.content

            elif api_choice == "ModelsLab":
                logger.info("Using ModelsLab API for image generation.")
                if not MODELSLAB_API_KEY:
                    raise gr.Error("Missing ModelsLab API Key (MODELSLAB_API_KEY).")

                ml_url = "https://modelslab.com/api/v6/realtime/text2img"
                ml_payload = {
                    "key": MODELSLAB_API_KEY,
                    "prompt": prompt,
                    "negative_prompt": common_negative_prompt,
                    "width": "1080",
                    "height": "1920",
                    "samples": 1,
                    "safety_checker": True,
                    "seed": None,
                    "base64": "no",
                    "webhook": None,
                    "track_id": None
                }
                ml_headers = {'Content-Type': 'application/json'}
                response = requests.post(ml_url, headers=ml_headers, json=ml_payload)
                if response.status_code != 200:
                    logger.error(f"ModelsLab API Error {response.status_code}: {response.text}")
                    raise gr.Error(f"ModelsLab API Error {response.status_code}: {response.text}")
                
                response_data = response.json()
                if response_data.get("status") == "success" and response_data.get("output"):
                    image_url = response_data["output"][0]
                    logger.info(f"Image URL received from ModelsLab: {image_url}")
                    image_response = requests.get(image_url, stream=True)
                    image_response.raise_for_status()
                    image_data = image_response.content
                else:
                    logger.error(f"ModelsLab API did not return a successful image: {response_data}")
                    raise gr.Error(f"Failed to get image from ModelsLab: {response_data.get('status', 'Unknown error')}")
            else:
                raise gr.Error(f"Invalid API choice: {api_choice}")

            if not image_data:
                raise gr.Error(f"No image data received from {api_choice} API.")

            try:
                image = Image.open(io.BytesIO(image_data))
                if image.mode in ('RGBA', 'LA'):
                    pass
                else:
                    image = image.convert("RGB")
                
                filename = f"image_{idx}.png"
                image.save(filename, format='PNG')
                
                images.append(image)
                image_filenames.append(filename)
                logger.info(f"Successfully generated and saved image {idx} using {api_choice}")
                
            except Exception as img_error:
                logger.error(f"Failed to process image from {api_choice}: {img_error}")
                raise gr.Error(f"Failed to decode image from {api_choice} API response.")

        return images, image_filenames
    except Exception as e:
        logger.error(f"Error in generate_images: {str(e)}")
        raise
def create_video(
    image_files: List[str],
    image_durations: List[float],
    audio_path: str = None,
    subtitles: List[ImageClip] = None,
    output_filename: str = "final_video.mp4",
    output_fps: int = 24, # Increased default FPS for smoother animations
) -> str:
    """Create video from images with specified durations, audio, and subtitles."""
    logger.info(f"Starting video creation for {output_filename} with {len(image_files)} images.")

    if not image_files:
        logger.error("No image files provided for video creation.")
        raise ValueError("No images to create video from.")
    if len(image_files) != len(image_durations):
        logger.error(
            f"Mismatch: {len(image_files)} images, {len(image_durations)} durations."
        )
        raise ValueError(
            "Number of image files must match number of image durations."
        )

    created_clips_to_close = []
    final_video_product = None

    try:
        processed_image_clips = []
        FADE_DURATION = 0.5  # Duration of the crossfade in seconds

        for img_path, duration in zip(image_files, image_durations):
            if not os.path.exists(img_path):
                logger.error(f"Image file not found: {img_path}")
                raise FileNotFoundError(f"Image file not found: {img_path}")

            # Create base image clip
            base_img_clip = ImageClip(img_path).set_duration(duration)
            
            # Apply standard transformations (resize and crop)
            transformed_clip = base_img_clip.resize(height=1920)
            transformed_clip = transformed_clip.crop(x_center=transformed_clip.w // 2, width=1080, height=1920)

            # Enhanced Ken Burns effect: faster zoom and pan
            final_zoom_factor = 1.35  # Zoom up to 135% (more zoom)
            target_w, target_h = 1080, 1920  # Target resolution

            def ken_burns_transform(get_frame, t):
                frame = get_frame(t)  # Original frame (1080x1920)
                pil_frame = Image.fromarray(frame)  # To PIL image

                # Faster zoom (more change per second)
                current_zoom = 1 + (final_zoom_factor - 1) * min(1.0, 2.0 * (t / duration))  # Up to 2x speed

                zoomed_w = int(target_w * current_zoom)  # New width
                zoomed_h = int(target_h * current_zoom)  # New height

                zoomed_pil_frame = pil_frame.resize((zoomed_w, zoomed_h), Image.LANCZOS)  # Resize
                zoomed_np_frame = np.array(zoomed_pil_frame)  # To numpy array

                # Pan from farther point (more noticeable movement)
                pan_x = int((zoomed_w - target_w) * min(1.0, 1.5 * (t / duration)))  # Increased pan speed
                pan_y = int((zoomed_h - target_h) * min(1.0, 1.5 * (t / duration)))

                # Return cropped frame
                return zoomed_np_frame[pan_y:pan_y + target_h, pan_x:pan_x + target_w]

            ken_burns_clip = transformed_clip.fl(ken_burns_transform, apply_to=['mask'])
            
            processed_image_clips.append(ken_burns_clip)

        if not processed_image_clips:
            logger.error("No video segments were created from images.")
            raise ValueError("Failed to create video segments.")

        # Assemble clips with crossfades
        if len(processed_image_clips) == 1:
            video_base = processed_image_clips[0]
        else:
            composited_clips_for_final_video = []
            current_timeline_pos = 0
            for i, clip in enumerate(processed_image_clips):
                # Ensure clip duration is not less than fade duration for proper fading
                # This check is simplified; more robust handling might be needed for very short clips
                actual_fade_duration = min(FADE_DURATION, clip.duration / 2 if clip.duration > 0 else 0.01)

                if i == 0:
                    clip_to_add = clip.set_start(0)
                    # Next clip will start actual_fade_duration earlier than this clip ends
                    current_timeline_pos = clip.duration - actual_fade_duration 
                else:
                    clip_to_add = clip.set_start(current_timeline_pos).crossfadein(actual_fade_duration)
                    # Advance timeline by the new clip's duration, minus the overlap
                    current_timeline_pos += clip.duration - actual_fade_duration
                
                composited_clips_for_final_video.append(clip_to_add)
            
            video_base = CompositeVideoClip(composited_clips_for_final_video, size=(1080, 1920))
            # The duration of video_base is now sum(durations) - (N-1)*actual_fade_duration

        created_clips_to_close.append(video_base)
        final_video_product = video_base

        if audio_path and os.path.exists(audio_path):
            audio_clip_instance = AudioFileClip(audio_path)
            created_clips_to_close.append(audio_clip_instance)
            
            # Ensure the visual part (video_base) is set to the audio's duration.
            # This will hold the last frame if the visual sequence is shorter.
            final_video_product = video_base.set_duration(audio_clip_instance.duration)

            if subtitles:
                logger.info(f"Adding {len(subtitles)} subtitle clips to video.")
                for sub_c in subtitles: created_clips_to_close.append(sub_c)
                # Use the duration-adjusted final_video_product as the base for subtitles
                final_video_product = CompositeVideoClip([final_video_product] + subtitles, size=(1080, 1920))
                created_clips_to_close.append(final_video_product)
            
            final_video_product = final_video_product.set_audio(audio_clip_instance)
        
        elif subtitles: # No audio, but subtitles exist
            logger.info(f"Adding {len(subtitles)} subtitle clips to video (no audio).")
            for sub_c in subtitles: created_clips_to_close.append(sub_c)
            # video_base duration is based on image sequence here
            final_video_product = CompositeVideoClip([video_base] + subtitles, size=(1080, 1920))
            created_clips_to_close.append(final_video_product)

        logger.info(f"Writing video to: {output_filename} with FPS: {output_fps}")
        final_video_product.write_videofile(
            output_filename,
            codec="libx264",
            fps=output_fps,
            audio_codec="aac",
            temp_audiofile=f"temp/temp-audio-{int(time.time())}.m4a",
            remove_temp=True,
            logger=None,
        )

        if os.path.exists(output_filename):
            logger.info(f"Successfully created video: {output_filename}")
            return output_filename
        else:
            logger.error(f"Video file {output_filename} not found after creation attempt.")
            raise ValueError("Failed to create video file")
            
    except Exception as e:
        logger.error(f"Error in create_video: {str(e)}")
        raise gr.Error(f"Error creating video: {str(e)}")
    finally:
        logger.debug(f"Cleaning up {len(created_clips_to_close)} clips in create_video.")
        # Close in reverse order of creation/dependency
        for clip_obj in reversed(created_clips_to_close):
            if clip_obj:
                try:
                    clip_obj.close()
                except Exception as e_close:
                    logger.warning(f"Error closing a clip of type {type(clip_obj)}: {e_close}")

async def process_inputs(story_text: str, prompts_text: str, voice: str, image_api_choice: str, existing_pil_images: List[Image.Image] = None, existing_image_filenames: List[str] = None):
    """
    Process story and image prompts to create a narrated video with scene alignment.
    """
    images = []
    image_filenames = []
    generated_audio_path = None # Renamed to avoid confusion
    
    try:
        logger.info("Starting input processing")
        
        # 1. Generate image prompts using $ separator
        logger.info("Parsing prompts using $ separator")
        prompt_list = [p.strip().strip('"') for p in prompts_text.split("$")]
        logger.info(f"Detected {len(prompt_list)} image prompts")

        # 2. Parse story into scenes using # separator
        logger.info("Splitting story into scenes using #")
        scene_list = [scene.strip() for scene in story_text.split("#") if scene.strip()]
        logger.info(f"Detected {len(scene_list)} story scenes")

        # 3. Check alignment
        if len(scene_list) > len(prompt_list):
            raise gr.Error("More story scenes than image prompts. Please match them.")

        # 4. Generate images or use existing ones
        if existing_image_filenames and len(existing_image_filenames) == len(prompt_list):
            logger.info(f"Using {len(existing_image_filenames)} existing images for video creation.")
            images = existing_pil_images # Assumes existing_pil_images corresponds to existing_image_filenames
            image_filenames = existing_image_filenames
        elif prompt_list and any(p.strip() for p in prompt_list): # Only generate if prompts are actually there
            logger.info(f"Starting image generation for {len(prompt_list)} prompts")
            images, image_filenames = generate_images(prompt_list, api_choice=image_api_choice)
            logger.info(f"Generated {len(image_filenames)} images")
        else: # No prompts and no existing images for prompts
            images, image_filenames = [], []


        # 5. Remove # before TTS
        story_cleaned = " ".join(scene_list)  # Flatten story
        logger.info("Generating audio from cleaned story text")
        generated_audio_path = await text_to_speech(story_cleaned, AVAILABLE_VOICES[voice])
        if not generated_audio_path:
            raise gr.Error("Failed to generate audio")
        logger.info(f"Generated audio at: {generated_audio_path}")

        # 6. Subtitles (optional)
        voice_id = voice.split(" - ")[0] if " - " in voice else voice
        if "Hindi" in voice_id:
            subtitles = None
            logger.info("Hindi voice detected — skipping subtitles")
        else:
            subtitles = generate_subtitles(generated_audio_path)

        # 7. Determine image_filenames_for_video and their durations
        image_filenames_for_video: List[str] = []
        image_durations: List[float] = []
        output_video_fps = 24 # Increased FPS for smoother animations

        temp_audio_clip_for_duration = AudioFileClip(generated_audio_path)
        total_audio_duration = temp_audio_clip_for_duration.duration
        temp_audio_clip_for_duration.close()

        num_scenes = len(scene_list)
        num_images_available = len(image_filenames)

        if num_images_available == 0 and total_audio_duration > 0:
            logger.error("No images were generated, cannot create video.")
            raise gr.Error("No images were generated. Cannot create video.")
        elif num_images_available == 0 and total_audio_duration == 0:
             logger.error("No images and no audio content.")
             raise gr.Error("No content (story, prompts) to generate video.")

        if num_scenes > 0:
            # Calculate durations based on word count per scene
            scene_word_counts = [len(scene.split()) for scene in scene_list]
            total_scene_words = sum(scene_word_counts)

            if total_scene_words == 0:
                # Fallback if somehow all scenes are empty after stripping
                logger.warning("Total scene word count is 0. Falling back to equal duration per scene.")
                duration_per_scene = total_audio_duration / num_scenes if num_scenes > 0 else 0
                image_durations = [duration_per_scene] * num_scenes
            else:
                # Distribute total audio duration based on word count proportion
                for count in scene_word_counts:
                    proportion = count / total_scene_words
                    image_durations.append(total_audio_duration * proportion)

            # Ensure total duration matches audio duration (handle potential floating point inaccuracies)
            # This might slightly adjust individual scene durations
            current_total_visual_duration = sum(image_durations)
            if current_total_visual_duration > 0:
                adjustment_factor = total_audio_duration / current_total_visual_duration
                image_durations = [d * adjustment_factor for d in image_durations]
            else: # Should only happen if total_audio_duration is 0
                 image_durations = [0.0] * len(image_durations)

            # Assign images to durations (one image per scene duration)
            for i in range(num_scenes):
                img_idx = min(i, num_images_available - 1) 
                image_filenames_for_video.append(image_filenames[img_idx])
        elif num_images_available > 0: # No scenes, but images exist
            duration_per_image = total_audio_duration / num_images_available
            image_filenames_for_video = image_filenames[:] # Use all images
            image_durations = [duration_per_image] * num_images_available

        # 8. Create final video
        os.makedirs("temp", exist_ok=True)
        video_output_filename = f"temp/final_video.mp4"

        video_path = create_video(
            image_files=image_filenames_for_video,
            image_durations=image_durations,
            audio_path=generated_audio_path,
            subtitles=subtitles,
            output_filename=video_output_filename,
            output_fps=output_video_fps
        )
        logger.info(f"Video processing completed. Output at: {video_path}")

        # Cleanup audio file
        if generated_audio_path and os.path.exists(generated_audio_path):
            os.unlink(generated_audio_path)
            logger.info(f"Cleaned up audio file: {generated_audio_path}")

        logger.info(f"Successfully completed video generation at {video_path}")
        import time
        # Wait up to 3 seconds for file to appear
        max_wait = 3
        waited = 0
        while not os.path.exists(video_path) and waited < max_wait:
            time.sleep(0.5)
            waited += 0.5
                
        file_exists = os.path.exists(video_path)
        logger.info(f"Final check, video path: {video_path}")
        logger.info(f"File exists after wait: {file_exists}")
        logger.info(f"File size (if exists): {os.path.getsize(video_path) if file_exists else 0}")
        return images, video_path, image_filenames # Return filenames for state update

    except Exception as e:
        for filename in image_filenames:
            if os.path.exists(filename):
                os.unlink(filename)
        if generated_audio_path and os.path.exists(generated_audio_path): # Use correct variable
            os.unlink(generated_audio_path)
        logger.error(f"Error in process_inputs: {str(e)}")
        raise gr.Error(str(e))
        
# Create the Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# Story to Video Generator")

    # State variables to hold generated images and their filenames
    image_pil_state = gr.State(value=[])
    image_filenames_state = gr.State(value=[])
    
    with gr.Row():
        with gr.Column():
            story_input = gr.Textbox(
                label="Enter your story for narration",
                lines=5,
                placeholder="Once upon a time..."
            )
            prompts_input = gr.Textbox(
                label='Enter image prompts (comma-separated, each in "")',
                lines=5,
                placeholder='"a castle in the forest", "a magical unicorn"'
            )
        
        with gr.Column():
            voice_select = gr.Dropdown(
                choices=list(AVAILABLE_VOICES.keys()),
                value="Male (American)",
                label="Select Voice"
            )
            image_api_select = gr.Radio(
                choices=["Cloudflare", "ModelsLab"],
                value="Cloudflare",
                label="Select Image Generation API",
                info="Ensure API keys are set for the chosen service."
            )
    
    generate_button = gr.Button("Generate Video")
    
    # Accordion for modifying images
    with gr.Accordion("Modify Generated Images", open=False):
        with gr.Row():
            image_select_dropdown = gr.Dropdown(
                label="Select Image to Replace (e.g., Image 1)", 
                interactive=True
            )
            replacement_image_upload = gr.File(
                label="Upload Replacement Image (9:16 recommended)", 
                type="filepath",
                file_types=["image"]
            )
        apply_replacement_button = gr.Button("Apply Replacement to Selected Image")
        replacement_status_text = gr.Textbox(label="Replacement Status", interactive=False)

    with gr.Row():
        # Changed gallery to use "pil" type instead of "filepath"
        gallery = gr.Gallery(label="Generated Images", type="pil")
        video_output = gr.Video(label="Generated Video")
    
    # Simplified the button click handler
    def generate_content_wrapper(story_text, prompts_text, voice, image_api_choice, current_pils, current_filenames):
        # This function is called by both buttons.
        # When called by generate_button, current_pils and current_filenames will be None.
        # When called by apply_replacement_and_generate, they will be the updated states.
        try:
            # Clear the temp folder at the beginning of any generation process
            clear_temp_folder()

            # If states are populated, it means we might be re-generating video after a replacement
            # or just re-running. process_inputs will decide if new images are needed.
            logger.info(f"generate_content_wrapper called. Current filenames in state: {len(current_filenames) if current_filenames else 0}")
            # Run the async function
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                # Pass current state to process_inputs
                # process_inputs will decide whether to generate new images or use existing ones
                pil_images_result, video_path_result, image_filenames_result = loop.run_until_complete(
                    process_inputs(story_text, prompts_text, voice, image_api_choice, current_pils, current_filenames)
                )
                
                # Removed the cleanup loop for image_*.png files.
                # These files will now persist and be overwritten by 
                # subsequent image generation or replacement.
                # The temporary audio file is cleaned up within process_inputs.
                # The temporary video file is handled by its own lifecycle.
                
                logger.info(f"Returning video path: {video_path_result}")
                import time
                
                # Wait up to 3 seconds for file to appear
                max_wait = 3
                waited = 0
                while not os.path.exists(video_path_result) and waited < max_wait:
                    time.sleep(0.5)
                    waited += 0.5
                
                file_exists = os.path.exists(video_path_result)
                logger.info(f"Video created at: {video_path_result}")
                logger.info(f"File exists after wait: {file_exists}")
                logger.info(f"File size (if exists): {os.path.getsize(video_path_result) if file_exists else 0}")
                
                # Update gallery, video output, and the states with the results
                return pil_images_result, video_path_result, pil_images_result, image_filenames_result
         
            finally:
                loop.close()
        except Exception as e:
            logger.error(f"Error in generate_content: {str(e)}")
            raise gr.Error(str(e))

    def handle_image_replacement(selected_image_label, uploaded_file, current_pils, current_filenames):
        if not selected_image_label or not uploaded_file:
            return current_pils, current_pils, current_filenames, "Please select an image from the dropdown and upload a replacement file."

        try:
            # selected_image_label is "Image X", convert to 0-based index
            selected_idx = int(selected_image_label.split(" ")[1]) - 1

            if not (0 <= selected_idx < len(current_pils)):
                return current_pils, current_pils, current_filenames, "Invalid image index selected."
            
            logger.info(f"Replacing image at index {selected_idx} with {uploaded_file.name}")

            # Load uploaded image
            uploaded_pil = Image.open(uploaded_file.name)

            # Process the uploaded PIL image (resize and crop to 1080x1920)
            # Convert PIL to NumPy array for ImageClip
            temp_clip = ImageClip(np.array(uploaded_pil))
            processed_pil = None
            try:
                transformed_clip = temp_clip.resize(height=1920)
                transformed_clip = transformed_clip.crop(x_center=transformed_clip.w // 2, width=1080, height=1920)
                processed_pil = Image.fromarray(transformed_clip.get_frame(0)) # Get frame at t=0
            finally: # Ensure clips are closed
                if temp_clip: temp_clip.close()
                if 'transformed_clip' in locals() and transformed_clip: transformed_clip.close()

            # Overwrite the original image file with the processed uploaded image
            original_filename_to_replace = current_filenames[selected_idx]
            processed_pil.save(original_filename_to_replace, format='PNG')
            logger.info(f"Saved processed replacement image to {original_filename_to_replace}")

            # Update the PIL images list for the gallery
            new_pils_list = list(current_pils)
            new_pils_list[selected_idx] = processed_pil
            
            # The current_filenames list (paths) doesn't change because we overwrite the file.
            # Return: gallery_pils, state_pils, state_filenames, status_text
            return new_pils_list, new_pils_list, current_filenames, f"Image {selected_idx+1} replaced successfully. Generating video..."
        except Exception as e:
            logger.error(f"Error replacing image: {e}")
            return current_pils, current_pils, current_filenames, f"Error: {str(e)}"

    # New wrapper for "Apply Replacement" button
    def apply_replacement_and_regenerate_video(selected_image_label, uploaded_file, 
                                               story_text, prompts_text, voice, image_api_choice, 
                                               current_pils_from_state, current_filenames_from_state):
        
        # Step 1: Handle the image replacement
        # handle_image_replacement returns: gallery_pils, state_pils, state_filenames, status_text
        gallery_pils_after_replacement, updated_pils_for_state, updated_filenames_for_state, status = handle_image_replacement(
            selected_image_label, uploaded_file, current_pils_from_state, current_filenames_from_state
        )

        if "Error:" in status: # If replacement failed, return current state and error status
            return gallery_pils_after_replacement, None, updated_pils_for_state, updated_filenames_for_state, status

        # Step 2: Trigger video generation with the updated images
        # generate_content_wrapper returns: pil_images_result, video_path_result, pil_images_result_for_state, image_filenames_result_for_state
        final_gallery_pils, video_path, final_pils_for_state, final_filenames_for_state = generate_content_wrapper(
            story_text, prompts_text, voice, image_api_choice, 
            updated_pils_for_state, # Pass the newly updated PIL list
            updated_filenames_for_state  # Pass the (potentially unchanged) filenames list
        )
        
        return final_gallery_pils, video_path, final_pils_for_state, final_filenames_for_state, status
    def update_image_selection_dropdown(pil_image_list_from_state):
        choices = [f"Image {i+1}" for i in range(len(pil_image_list_from_state))]
        return gr.update(choices=choices, value=choices[0] if choices else None)
    
    generate_button.click(
        fn=generate_content_wrapper,
        # Pass None for current_pils and current_filenames to force regeneration of images
        inputs=[story_input, prompts_input, voice_select, image_api_select, gr.State(None), gr.State(None)],
        outputs=[gallery, video_output, image_pil_state, image_filenames_state]
    )

    apply_replacement_button.click(
        fn=apply_replacement_and_regenerate_video,
        inputs=[
            image_select_dropdown, replacement_image_upload, 
            story_input, prompts_input, voice_select, image_api_select, # For video generation
            image_pil_state, image_filenames_state # Current states
        ],
        outputs=[gallery, video_output, image_pil_state, image_filenames_state, replacement_status_text]
    )

    image_pil_state.change( # When new images are generated and state updates
        fn=update_image_selection_dropdown,
        inputs=[image_pil_state],
        outputs=[image_select_dropdown]
    )

if __name__ == "__main__":
    demo.launch()
