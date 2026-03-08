import streamlit as st
import subprocess
import json
import tempfile
import os
import cv2
from PIL import Image
import sys  # for current Python executable

# Configure page
st.set_page_config(
    page_title="Video Analysis Dashboard",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark theme
st.markdown("""
<style>
    .main { background-color: #0e1117; color: #fafafa; }
    .stApp { background-color: #0e1117; }
    .stSelectbox > div > div, .stFileUploader > div > div { background-color: #262730; color: #fafafa; }
    .stButton > button { background-color: #ff4b4b; color: white; border: none; border-radius: 0.5rem; padding: 0.5rem 1rem; font-weight: bold; }
    .stButton > button:hover { background-color: #ff6b6b; }
    .result-card { background-color: #262730; padding: 1rem; border-radius: 0.5rem; margin: 1rem 0; border-left: 4px solid #ff4b4b; }
    .success-card { border-left-color: #00d4aa; }
    .warning-card { border-left-color: #ffa726; }
    .error-card { border-left-color: #f44336; }
    .metric-card { background-color: #1e1e2e; padding: 1rem; border-radius: 0.5rem; text-align: center; margin: 0.5rem; }
</style>
""", unsafe_allow_html=True)

def save_uploaded_file(uploaded_file):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            return tmp_file.name
    except Exception as e:
        st.error(f"Error saving file: {str(e)}")
        return None

def run_analysis(video_path, model_type):
    """Run the test script using the same Python environment as Streamlit"""
    try:
        script_path = "violence_test.py" if model_type == "Violence Detection" else "stampede_test.py"
        
        python_executable = sys.executable

        result = subprocess.run(
            [python_executable, script_path, video_path],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        if result.returncode != 0:
            return {"error": f"Script execution failed:\n{result.stderr}", "success": False}
        
        try:
            return json.loads(result.stdout.strip())
        except json.JSONDecodeError:
            return {"error": f"Failed to parse script output:\n{result.stdout}", "success": False}

    except subprocess.TimeoutExpired:
        return {"error": "Analysis timed out after 5 minutes", "success": False}
    except Exception as e:
        return {"error": f"Error running analysis: {str(e)}", "success": False}

def get_video_preview(video_path, max_frames=10):
    try:
        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_count = 0
        while len(frames) < max_frames:
            ret, frame = cap.read()
            if not ret: break
            if frame_count % 5 == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_resized = cv2.resize(frame_rgb, (200, 150))
                frames.append(Image.fromarray(frame_resized))
            frame_count += 1
        cap.release()
        return frames
    except Exception as e:
        st.error(f"Error generating video preview: {str(e)}")
        return []

def display_results(result, model_type):
    if not result.get("success", False):
        st.markdown(f'<div class="result-card error-card"><h3>Analysis Failed</h3><p>{result.get("error","Unknown error")}</p></div>', unsafe_allow_html=True)
        return
    
    classification = result.get("final_classification", "")
    card_class = "warning-card" if "Violence" in classification or "Stampede" in classification else "success-card"
    
    st.markdown(f'<div class="result-card {card_class}"><h2>{classification}</h2><p>Model: {model_type}</p></div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f'<div class="metric-card"><h3>{result.get("total_frames",0)}</h3><p>Total Frames Analyzed</p></div>', unsafe_allow_html=True)

    confidence = result.get("average_confidence", 0)

    # ✅ Only show confidence details if >= 0.8
    if confidence >= 0.68:
        with col2:
            st.markdown(f'<div class="metric-card"><h3>{confidence:.4f}</h3><p>Average Confidence Score</p></div>', unsafe_allow_html=True)
        with col3:
            st.markdown(f'<div class="metric-card"><h3>{confidence*100:.1f}%</h3><p>Confidence Percentage</p></div>', unsafe_allow_html=True)

def main():
    st.title("Video Analysis Dashboard")
    st.markdown("Upload a video and select a model to analyze for violence detection or crowd management.")
    
    with st.sidebar:
        st.header("Controls")
        uploaded_file = st.file_uploader("Choose a video file", type=['mp4','avi','mov','mkv','wmv'])
        model_type = st.selectbox("Select Analysis Model", ["Violence Detection", "Crowd Management"])
        analyze_button = st.button("Start Analysis", type="primary")
    
    col1, col2 = st.columns([1,1])
    
    with col1:
        st.header("Video Preview")
        video_path = None
        if uploaded_file:
            video_path = save_uploaded_file(uploaded_file)
            if video_path:
                st.success(f"File uploaded: {uploaded_file.name}")
                st.info(f"File size: {uploaded_file.size/(1024*1024):.2f} MB")
                with st.spinner("Generating video preview..."):
                    preview_frames = get_video_preview(video_path)
                if preview_frames:
                    st.write("*Preview frames:*")
                    cols = st.columns(3)
                    for i, frame in enumerate(preview_frames[:6]):
                        with cols[i % 3]:
                            st.image(frame, caption=f"Frame {i+1}")
                else:
                    st.warning("Could not generate video preview")
        else:
            st.info("Please upload a video file to get started")
    
    with col2:
        st.header("Analysis Results")
        if uploaded_file and analyze_button:
            if not video_path:
                st.error("Failed to save uploaded file")
            else:
                with st.spinner("Analyzing video... This may take a few minutes."):
                    result = run_analysis(video_path, model_type)
                display_results(result, model_type)
                try: os.unlink(video_path)
                except: pass
        else:
            st.info("Upload a video and click 'Start Analysis' to see results here")
    
    st.markdown("---")
    st.markdown('<div style="text-align:center;color:#666;"><p>Video Analysis Dashboard | Built with Streamlit</p></div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
