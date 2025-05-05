#!/usr/bin/env python3
"""
Banana Quality Classification using MobileNetV2 on Jetson Nano
- Captures images every 15 seconds
- Classifies banana quality (good, intermediate, bad)
- Controls GPIO pins based on classification results
"""

import os
import time
import cv2
import numpy as np
import tensorflow as tf
import Jetson.GPIO as GPIO
from datetime import datetime

# GPIO Setup
GPIO.setmode(GPIO.BOARD)
GOOD_PIN = 11       # GPIO pin for good banana indicator
INTERMEDIATE_PIN = 13  # GPIO pin for intermediate banana indicator
BAD_PIN = 15        # GPIO pin for bad banana indicator

# Setup GPIO pins as outputs
output_pins = [GOOD_PIN, INTERMEDIATE_PIN, BAD_PIN]
for pin in output_pins:
    GPIO.setup(pin, GPIO.OUT)
    GPIO.output(pin, GPIO.LOW)  # Initialize all pins to LOW

# Define image parameters
IMG_WIDTH = 224
IMG_HEIGHT = 224
CAPTURE_INTERVAL = 15  # seconds

# Create directory for saving captured images
SAVE_DIR = "captured_images"
os.makedirs(SAVE_DIR, exist_ok=True)

def load_model():
    """Load and prepare the MobileNetV2 model for banana classification"""
    print("Loading MobileNetV2 model...")
    
    # Load the base MobileNetV2 model (pre-trained on ImageNet)
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze the base model layers
    base_model.trainable = False
    
    # Add custom classification head
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(3, activation='softmax')  # 3 classes: good, intermediate, bad
    ])
    
    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("Model loaded successfully!")
    
    # In a real implementation, you would load trained weights here
    # model.load_weights('banana_classifier_weights.h5')
    
    return model

def preprocess_image(image):
    """Preprocess image for the MobileNetV2 model"""
    image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def classify_banana(model, image):
    """Classify banana quality using the trained model"""
    preprocessed_image = preprocess_image(image)
    prediction = model.predict(preprocessed_image)[0]
    class_idx = np.argmax(prediction)
    confidence = prediction[class_idx]
    
    # Map class index to label
    labels = ["good", "intermediate", "bad"]
    label = labels[class_idx]
    
    return label, confidence

def update_gpio(label):
    """Set GPIO pins based on classification result"""
    # Reset all pins
    for pin in output_pins:
        GPIO.output(pin, GPIO.LOW)
    
    # Set appropriate pin HIGH
    if label == "good":
        GPIO.output(GOOD_PIN, GPIO.HIGH)
    elif label == "intermediate":
        GPIO.output(INTERMEDIATE_PIN, GPIO.HIGH)
    elif label == "bad":
        GPIO.output(BAD_PIN, GPIO.HIGH)

def capture_image(camera):
    """Capture an image from the camera"""
    print("Capturing image...")
    ret, frame = camera.read()
    if not ret:
        print("Failed to capture image!")
        return None
    
    # Save image with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{SAVE_DIR}/banana_{timestamp}.jpg"
    cv2.imwrite(filename, frame)
    print(f"Image saved as {filename}")
    
    return frame

def main():
    try:
        # Initialize camera (gstreamer for Jetson compatibility)
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            print("Error: Could not open camera.")
            return
        
        # Allow camera to initialize
        print("Initializing camera...")
        time.sleep(2)
        
        # Load the classification model
        model = load_model()
        
        print("Starting banana quality monitoring...")
        while True:
            # Capture image
            frame = capture_image(camera)
            if frame is None:
                continue
            
            # Classify banana
            label, confidence = classify_banana(model, frame)
            print(f"Classification: {label} (confidence: {confidence:.2f})")
            
            # Update GPIO pins
            update_gpio(label)
            
            # Wait for next capture
            print(f"Waiting {CAPTURE_INTERVAL} seconds until next capture...")
            time.sleep(CAPTURE_INTERVAL)
    
    except KeyboardInterrupt:
        print("Program terminated by user")
    
    finally:
        # Clean up
        camera.release()
        GPIO.cleanup()
        print("Resources released, program exited")

if __name__ == "__main__":
    main()