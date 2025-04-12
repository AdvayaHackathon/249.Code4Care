# backend/voice_model.py

import speech_recognition as sr

def get_symptoms_from_audio(audio_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio = recognizer.record(source)
        text = recognizer.recognize_google(audio)

    keywords = ["chest pain", "bleeding", "fainting", "rash", "fever", "swelling"]
    severity = "Mild"

    for k in keywords:
        if k in text.lower():
            if k in ["chest pain", "bleeding", "fainting"]:
                severity = "Critical"
            elif k in ["rash", "swelling"]:
                severity = "Moderate"
    return text, severity
