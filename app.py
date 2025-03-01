import re
from flask import Flask, request, render_template, url_for
import language_tool_python
from pydub import AudioSegment, silence
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import numpy as np
import os
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
import librosa
import torch
import whisper

# Load Whisper model
whisper_model = whisper.load_model("large")
# Load audio
def load_audio(file_path):
    return AudioSegment.from_file(file_path)

# Load emotion detection model
emotion_model_id = "firdhokk/speech-emotion-recognition-with-openai-whisper-large-v3"
emotion_model = AutoModelForAudioClassification.from_pretrained(emotion_model_id)
feature_extractor = AutoFeatureExtractor.from_pretrained(emotion_model_id, do_normalize=True)
id2label = emotion_model.config.id2label

# Flask app setup
app = Flask(__name__)

# Filler words list
filler_words =  [
    "uh-huh", "mm-hmm", "ah-ha", "uh-oh", "hmm-mmm", "uh-huh-huh", "hmm-yeah",
    "oh-um", "uh-yeah", "tsk-tsk", "huh-uh", "eh-heh", "heh-heh", "wel-", "so-",
    "uh-huh-uh", "mmm-hmm", "uh-er", "mmm-ah", "uh-um", "huh-hmm", "eh-um",
    "ugh-huh", "ah-huh", "oh-oh", "ah", "hmm.", "mm", "Um", "Ah", "Hmm", "Er",
    "uh", "Yeah", "Nah", "Oop", "Aye", "Eh", "Hey", "Yo", "um", "uh", "ah",
    "er", "hmm", "mmm", "eh", "oh", "ehm", "huh", "aha", "ahh", "errr", "mmmm",
    "ehhh", "eh", "naah", "haa", "lah", "mah", "ahem", "tsk", "pfft", "grr",
    "uhhh", "ummm", "shh", "ohhh", "whoa", "ahhh", "hmmm", "duh", "meh", "yawn",
    "aah", "uhmmm", "uhhn", "hng", "err", "hrrmm", "ahum", "haaah", "oops", "eww",
    "ugh", "nah", "hmmph", "uh-uh", "aiyah", "anoh", "eto", "ano", "arre", "huff",
    "phew", "argh", "hngh", "tch", "kch", "aaah", "ahhhhh", "ummmm", "ehhhhh",
    "lah", "leh", "cha", "che", "ani", "mmm-huh", "ahm", "hah"
]

# Preprocess audio for emotion detection
def preprocess_audio(audio_path, feature_extractor, max_duration=30.0):
    audio_array, sampling_rate = librosa.load(audio_path, sr=feature_extractor.sampling_rate)
    max_length = int(feature_extractor.sampling_rate * max_duration)
    if len(audio_array) > max_length:
        audio_array = audio_array[:max_length]
    else:
        audio_array = np.pad(audio_array, (0, max_length - len(audio_array)))
    inputs = feature_extractor(
        audio_array,
        sampling_rate=feature_extractor.sampling_rate,
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
    )
    return inputs

# Predict emotion
def predict_emotion(audio_path, model, feature_extractor, id2label, max_duration=30.0):
    inputs = preprocess_audio(audio_path, feature_extractor, max_duration)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_id = torch.argmax(logits, dim=-1).item()
    predicted_label = id2label[predicted_id]
    return predicted_label

# Process transcription for filler words
def process_transcription(text):
    refined_text = re.sub(r'\b(?:' + '|'.join(filler_words) + r')\b', '', text, flags=re.IGNORECASE).strip()
    feedback = "Good to go!" if text == refined_text else "Filler words removed. Keep practicing!"
    return refined_text, feedback

# Correct grammar
def correct_grammar(text):
    tool = language_tool_python.LanguageTool('en-US')
    matches = tool.check(text)
    corrected_text = language_tool_python.utils.correct(text, matches)
    return corrected_text

# Calculate pauses in audio
def calculate_pauses(audio, min_silence_len=500, silence_thresh=-40):
    pauses = silence.detect_silence(audio, min_silence_len=min_silence_len, silence_thresh=silence_thresh)
    return [(start, end) for start, end in pauses]

# Detect long pauses
def detect_long_pauses(pauses, threshold):
    long_pauses = []
    long_pause_timestamps = []
    for start, end in pauses:
        duration = end - start
        if duration > threshold:
            long_pauses.append((start, end))
            long_pause_timestamps.append((start / 1000, end / 1000))
    return long_pauses, long_pause_timestamps
def format_long_pause_details(long_pause_timestamps):
    """
    Format the long pause timestamps into HTML-formatted details
    
    Args:
        long_pause_timestamps (list): List of tuples containing (start, end) times in seconds
    
    Returns:
        str: HTML-formatted string containing pause details
    """
    if not long_pause_timestamps:
        return "<p>No significant long pauses detected.</p>"
    
    details = "<ul>"
    for start, end in long_pause_timestamps:
        duration = end - start
        details += f"<li>Pause from {start:.2f}s to {end:.2f}s (Duration: {duration:.2f}s)</li>"
    details += "</ul>"
    
    # Add summary information
    total_pauses = len(long_pause_timestamps)
    avg_duration = sum((end - start) for start, end in long_pause_timestamps) / total_pauses
    
    summary = f"""
    <p><strong>Summary:</strong></p>
    <ul>
        <li>Total number of significant pauses: {total_pauses}</li>
        <li>Average pause duration: {avg_duration:.2f} seconds</li>
    </ul>
    """
    
    return summary + details

# Plot pauses
def plot_pauses(pause_durations, long_pauses, threshold):
    if not pause_durations:  # Check if pause_durations is empty
        return None  # Return None to indicate no plot
    
    bin_interval = 250  # Adjust granularity as needed
    bins = np.arange(0, max(pause_durations) + bin_interval, bin_interval)

    plt.figure(figsize=(7, 4))  
    plt.hist(
        pause_durations,
        bins=bins,
        color='skyblue',
        edgecolor='black',
        alpha=0.7,
        label="All Pauses"
    )
    long_pause_durations = [end - start for start, end in long_pauses]
    plt.hist(
        long_pause_durations,
        bins=bins,
        color='orange',
        edgecolor='black',
        alpha=0.7,
        label="Long Pauses"
    )
    plt.axvline(threshold, color='red', linestyle='dashed', linewidth=1.5, label=f"Threshold: {threshold} ms")
    plt.title("Pause Durations in Audio")
    plt.xlabel("Length of Pauses (ms)")
    plt.ylabel("Number of Pauses")
    plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels
    plt.legend(loc="upper right")
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()  # Adjust layout to prevent cutoff

    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    return base64.b64encode(buf.getvalue()).decode('utf-8')


# Tone graph
def plot_emotions_probabilities(probabilities, id2label):
    labels = [id2label[i] for i in range(len(probabilities))]
    
    # Create a figure with a wider aspect ratio
    fig, ax = plt.subplots(figsize=(8, 6))

    # Create color map for emotions
    emotion_colors = {
        'happy': '#FFD700',    # Gold
        'sad': '#4169E1',      # Royal Blue
        'angry': '#DC143C',    # Crimson
        'neutral': '#808080',  # Gray
        'disgust': '#32CD32',  # Lime Green
        'fearful': '#800080',  # Purple
        'surprised': '#FF69B4', # Hot Pink
        'calm': '#87CEEB'      # Sky Blue
    }
    
    colors = [emotion_colors[label.lower()] for label in labels]

    wedges, texts, autotexts = ax.pie(probabilities, 
                                     colors=colors,
                                     autopct='%1.1f%%',
                                     startangle=90,
                                     pctdistance=0.85,
                                     wedgeprops={'edgecolor': 'white', 'linewidth': 2})

    # Add a title with the dominant emotion
    dominant_idx = np.argmax(probabilities)
    dominant_emotion = labels[dominant_idx]
    dominant_prob = probabilities[dominant_idx] * 100
    plt.title(f"Dominant Emotion: {dominant_emotion} ({dominant_prob:.1f}%)", 
             pad=20, 
             fontsize=14, 
             fontweight='bold')

    # Enhance legend
    legend_labels = [f"{labels[i]} ({probabilities[i]*100:.1f}%)" for i in range(len(labels))]
    plt.legend(wedges, legend_labels, 
              title="Emotions Distribution",
              loc="center left",
              bbox_to_anchor=(1, 0.5),
              fontsize=10)

    plt.axis('equal')
    plt.tight_layout()

    # Save to BytesIO
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=300)
    buf.seek(0)
    plt.close()
    
    return base64.b64encode(buf.getvalue()).decode('utf-8'), dominant_emotion, dominant_prob
def get_grammar_feedback(text):
    tool = language_tool_python.LanguageTool('en-US')
    matches = tool.check(text)
    return len(matches)  # Returns only the number of grammar errors


def generate_feedback(filler_count, grammar_count, long_pauses, predicted_emotion, probabilities):
    feedback = "<h2>Feedback Summary</h2>"

    # Filler Words Section
    if filler_count > 2:
        feedback += f"""
        <h3>1. Filler Words</h3>
        <p><strong>Observation:</strong> You used {filler_count} filler words. This indicates a slight dependency on fillers during your speech.</p>
        <p><strong>Feedback:</strong> Reducing filler words can enhance your speech clarity and make you appear more confident. Practice deliberate pauses instead of using fillers.</p>
        <p><strong>Why Minimal Filler Words Matter?:</strong> Excessive fillers may distract your audience, but speaking concisely ensures better engagement and strengthens your credibility.</p>
        """
    else:
        feedback += f"""
        <h3>1. Filler Words</h3>
        <p><strong>Observation:</strong> You used only {filler_count} filler word(s), which is excellent!</p>
        <p><strong>Feedback:</strong> Great job! Your speech is clear and focused, demonstrating confidence and preparation. Minimal fillers make your delivery polished and professional.</p>
        <p><strong>Why Minimal Filler Words Matter?:</strong> Limited use of fillers keeps your message sharp, helping you maintain your audience's attention effortlessly.</p>
        """

    # Long Pauses Section
    if long_pauses:
        feedback += f"""
        <h3>2. Long Pauses</h3>
        <p><strong>Observation:</strong> You had {len(long_pauses)} long pauses, which could disrupt the flow of your speech.</p>
        <p><strong>Feedback:</strong> Avoiding long pauses helps maintain a steady rhythm and keeps the audience engaged. Use shorter, deliberate pauses to emphasize key points instead.</p>
        <p><strong>Why Avoiding Long Pauses Is Good?:</strong> Long pauses can cause your audience to lose focus, but controlled pauses help emphasize critical ideas effectively.</p>
        """
    else:
        feedback += """
        <h3>2. Long Pauses</h3>
        <p><strong>Observation:</strong> No excessive long pauses were detected. Your speech maintained a steady and engaging flow.</p>
        <p><strong>Feedback:</strong> Well done! Using pauses effectively makes your delivery smooth and ensures that your audience remains attentive throughout.</p>
        <p><strong>Why Avoiding Long Pauses Is Good?:</strong> Steady pacing with well-timed pauses enhances the natural rhythm of your speech, making it more impactful.</p>
        """

    # Grammar Mistakes Section
    if grammar_count == 0:
        feedback += """
        <h3>3. Grammar</h3>
        <p><strong>Observation:</strong> No grammar mistakes were detected! Your speech was well-structured and grammatically sound.</p>
        <p><strong>Feedback:</strong> Excellent job! Proper grammar enhances clarity and makes your message more professional and credible.</p>
        <p><strong>Why Good Grammar Matters?:</strong> Well-structured sentences improve comprehension and strengthen your communication skills.</p>
        """
    elif grammar_count <= 3:
        feedback += f"""
        <h3>3. Grammar</h3>
        <p><strong>Observation:</strong> Your speech contained {grammar_count} minor grammatical mistakes.</p>
        <p><strong>Feedback:</strong> Minor grammar errors don't greatly impact clarity, but refining them can improve fluency. Reviewing sentence structure and practicing common grammar rules can help.</p>
        <p><strong>Why Good Grammar Matters?:</strong> Correct grammar ensures your message is clear and polished, making your speech more engaging.</p>
        """
    else:
        feedback += f"""
        <h3>3. Grammar</h3>
        <p><strong>Observation:</strong> Your speech had {grammar_count} grammar mistakes, which may have affected clarity.</p>
        <p><strong>Feedback:</strong> Improving grammar usage will enhance the effectiveness of your speech. Consider reviewing common mistakes and practicing structured speaking exercises.</p>
        <p><strong>Why Good Grammar Matters?:</strong> Clear and grammatically correct speech ensures that your audience understands your message without confusion.</p>
        """

    # Enhanced Emotion Section
    emotion_categories = {
        'positive': ['happy', 'calm', 'surprised'],
        'neutral': ['neutral'],
        'negative': ['sad', 'angry', 'disgust', 'fearful']
    }

    # Get the dominant emotion and its probability
    emotions = ["sad", "happy", "angry", "neutral", "disgust", "fearful", "surprised", "calm"]
    dominant_idx = np.argmax(probabilities)
    dominant_emotion = predicted_emotion
    dominant_prob = probabilities[dominant_idx] * 100

    # Determine emotion category
    emotion_category = next((cat for cat, emotions in emotion_categories.items() 
                           if dominant_emotion.lower() in emotions), 'neutral')

    # Generate emotion feedback based on category and specific emotion
    emotion_feedback = f"""
    <h3>4. Emotion Analysis</h3>
    <p><strong>Primary Emotion Detected:</strong> {predicted_emotion} ({dominant_prob:.1f}%)</p>
    """

    if emotion_category == 'positive':
        emotion_feedback += f"""
        <p><strong>Analysis:</strong> Your speech conveys {dominant_emotion.lower()} emotions, which is excellent! This positive tone helps create an engaging and welcoming atmosphere.</p>
        <p><strong>Impact:</strong> Positive emotions like {dominant_emotion.lower()} help build rapport with your audience and make your message more memorable.</p>
        <p><strong>Suggestions:</strong> Continue maintaining this positive energy while ensuring it matches your content appropriately.</p>
        """
    elif emotion_category == 'neutral':
        emotion_feedback += f"""
        <p><strong>Analysis:</strong> Your speech maintains a neutral tone, which can be appropriate for formal or professional contexts.</p>
        <p><strong>Impact:</strong> A neutral tone helps deliver information clearly without emotional bias, but consider adding some warmth when appropriate.</p>
        <p><strong>Suggestions:</strong> While neutrality is good, try incorporating subtle variations in tone to maintain audience engagement.</p>
        """
    else:  # negative
        emotion_feedback += f"""
        <p><strong>Analysis:</strong> Your speech shows predominantly {dominant_emotion.lower()} emotions, which might not be optimal depending on your message.</p>
        <p><strong>Impact:</strong> {dominant_emotion.title()} tones can make your message feel heavy or unapproachable, potentially affecting audience engagement.</p>
        <p><strong>Suggestions:</strong> Unless the content specifically requires this tone, try to:
            <ul>
                <li>Incorporate more positive vocal variations like happy,calm or neutral</li>
                <li>Practice delivering with a more balanced emotional tone</li>
                <li>Use breathing exercises before speaking to manage emotional expression</li>
            </ul>
        </p>
        """

    feedback += emotion_feedback
    return feedback

@app.route('/')
def home():
    return render_template('home.html')
@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part'
        file = request.files['file']
        if file.filename == '':
            return 'No selected file'
            
        file_path = f"uploads/{file.filename}"
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        try:
            file.save(file_path)
        except Exception as e:
            return f"File upload failed: {str(e)}"

        # Transcribe the audio
        result = whisper_model.transcribe(file_path)
        original_text = result["text"]

        # Process transcription and grammar correction
        refined_text, filler_feedback = process_transcription(original_text)
        corrected_text = correct_grammar(refined_text)

        # Count grammar mistakes correctly
        grammar_count = get_grammar_feedback(original_text)

        # Emotion analysis
        inputs = preprocess_audio(file_path, feature_extractor)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        emotion_model.to(device)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        
        with torch.no_grad():
            outputs = emotion_model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
        
        # Get enhanced emotion visualization and analysis
        pie_chart_url, dominant_emotion, dominant_prob = plot_emotions_probabilities(probabilities, id2label)

        # Pause analysis
        audio = load_audio(file_path)
        pauses = calculate_pauses(audio)
        pause_durations = [end - start for start, end in pauses]

        if pause_durations:
            threshold = max(1000, np.mean(pause_durations) + 1.5 * np.std(pause_durations))
            long_pauses, long_pause_timestamps = detect_long_pauses(pauses, threshold)
            pause_histogram_url = plot_pauses(pause_durations, long_pauses, threshold)
            long_pause_details = format_long_pause_details(long_pause_timestamps)
        else:
            long_pauses = []
            pause_histogram_url = None
            long_pause_details = "<p>No significant pauses detected.</p>"

        # Generate feedback
        filler_count = len(re.findall(r'\b(?:' + '|'.join(filler_words) + r')\b', original_text, flags=re.IGNORECASE))
        feedback_summary = generate_feedback(filler_count, grammar_count, long_pauses, dominant_emotion, probabilities)

        return render_template('feedback.html',
            original_text=original_text,
            refined_text=refined_text,
            grammar_corrected_text=corrected_text,
            grammar_count=grammar_count,
            long_pause_details=long_pause_details,
            plot_url=pause_histogram_url,
            pie_chart_url=pie_chart_url,
            dominant_emotion=dominant_emotion,
            dominant_prob=dominant_prob,
            feedback_summary=feedback_summary
        )

    return render_template('upload.html')

if __name__ == "__main__":
    app.run(debug=True)

