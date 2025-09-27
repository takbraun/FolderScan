# FolderScan
NSFW Content Scanner & Directory Cleanser
This is a powerful, web-based tool designed to recursively scan directories for Not-Safe-For-Work (NSFW) content, including both images and videos. It uses a sophisticated two-pass AI system to accurately identify and quarantine explicit and suggestive material, helping you to cleanse your media collections efficiently.

The application is built with a user-friendly web interface that allows for easy configuration of directories and sensitivity settings, providing real-time feedback on the scanning process.

Features
Web-Based Interface: Easy-to-use UI for starting, stopping, and monitoring scans from your browser.

GPU Acceleration: Automatically utilizes NVIDIA GPUs via CUDA for significantly faster processing times.

Two-Pass Scanning System:

Primary Nudity Scan: A highly accurate model (nudenet) identifies explicitly NSFW content.

Secondary Suggestive Scan: An optional second pass uses a heuristic-based classifier to catch more nuanced content like lingerie and suggestive poses that the primary model might miss.

Adjustable Sensitivity: Fine-tune the sensitivity of both the primary and secondary scanners with easy-to-use sliders and presets (Strict, Balanced, Relaxed).

Recursive Scanning: Deep-scans all subdirectories within a target folder to ensure no file is missed.

Safe Quarantine: Automatically moves detected files to a separate "quarantine" directory, preserving the original folder structure.

Prerequisites
Before you begin, ensure you have the following installed on your system:

Python 3.8+

pip (Python package installer)

(Recommended) An NVIDIA GPU with the appropriate CUDA drivers installed for significant performance improvements. The application will fall back to CPU mode if a compatible GPU is not detected.

Installation
Clone the Repository or Download the Files:

Place the app.py file in a dedicated directory for the project.

Create a Virtual Environment:

It is highly recommended to use a virtual environment to keep dependencies isolated.

Open your terminal in the project directory and run:

python3 -m venv venv

Activate the Virtual Environment:

On Linux/macOS:

source venv/bin/activate

On Windows:

venv\Scripts\activate

Install Required Libraries:

With your virtual environment active, install all the necessary libraries by running:

pip install flask flask_cors numpy opencv-python Pillow torch torchvision psutil nudenet

Usage
Activate Your Virtual Environment (if it's not already active):

source venv/bin/activate

Run the Application:

Execute the main Python script from your terminal:

python3 app.py

The terminal will show startup messages, including whether it's using the GPU or CPU.

Access the Web Interface:

Once the server is running, open your web browser and navigate to:
http://127.0.0.1:8000

Configure and Run a Scan:

Use the "Browse" buttons to select your Source Directory (the folder you want to clean) and your NSFW Quarantine Directory (where flagged files will be moved).

Adjust the sensitivity sliders and enable the "Suggestive Content Scan" if desired.

Click the "Start Directory Cleansing" button to begin.

Monitor the progress in both the UI and the terminal.

Disclaimer
This tool relies on AI models for detection, and no model is 100% perfect. It is possible for the scanner to produce both false positives (flagging a safe file) and false negatives (missing a NSFW file). It is recommended to manually review the quarantined files before deletion. The sensitivity sliders can be adjusted to help find the right balance for your needs.

Troubleshooting / FAQ
Why does the scanner report more files than I see in my folder?

This is normal behavior and means the scanner is working correctly. Your computer's file explorer (like Finder on macOS or File Explorer on Windows) often hides certain files by default to avoid clutter. This can include:

System Files: thumbs.db (Windows), .DS_Store (macOS).

Hidden Files and Folders: Any file or folder name that starts with a dot (e.g., .thumbnails) is hidden on Linux and macOS.

The NSFW scanner is designed to be thorough and will find all media files that match the supported extensions, regardless of whether they are hidden or not. This ensures that no potentially NSFW content is missed.

