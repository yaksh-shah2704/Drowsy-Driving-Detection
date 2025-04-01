<h2># Drowsy-Driving-Detection</h2>


<i><b>This project is designed to detect drowsy driving using computer vision and machine learning techniques. It uses **OpenCV**, **Dlib**, and **pygame** to analyze the driver's face and monitor their eye movement. If the system detects that the driver is becoming drowsy, it sends an alert (with sound and visual indicators).</i></b>

<h3>## Features</h3>

- Detects the driver's face and eyes in real-time using **OpenCV** and **Dlib**. <br>
- Calculates **Eye Aspect Ratio (EAR)** to determine drowsiness based on eye movement. <br>
- Plays an alert sound when the driver is detected to be drowsy (beep sound). <br>
- Visual indicators (alert text) displayed on the screen when the driver is drowsy. <br>

<h3>## Requirements</h3>

- Python 3.x <br>
- Install the required libraries by running:<br>

<b>```bash<b><br>
<i>pip install opencv-python imutils dlib pygame scipy</i>

<h3>Files Required <smmall>(both are placed in the project directory)</smmall></h3> 
<b>
shape_predictor_68_face_landmarks.dat: A pre-trained Dlib model for detecting facial landmarks. <br>
beep.mp3: The sound file to play when drowsiness is detected.</b>

<h4>Setup</h4>
<ol>
    <li>Clone the repository to your local machine:</li>
    git clone https://github.com/yaksh-shah2704/Drowsy-Driving-Detection.git<br>
    cd Drowsy-Driving-Detection
    <li>Make sure to have the shape_predictor_68_face_landmarks.dat file in the same directory as your script.</li>
    <li>Install the required dependencies (as mentioned in the "Requirements" section).</li>
</ol>

<h4>Flow of the Program</h4>
<ol>
    <li>Capture video feed from the webcam.</li>
    <li>Convert the frames to grayscale for faster processing.</li>
    <li>Detect faces using Dlib's face detector.</li>
    <li>Extract the eye landmarks and calculate the Eye Aspect Ratio (EAR).</li>
    <li>If the EAR falls below a threshold, it indicates the person is drowsy.</li>
    <li>Show an alert message on the screen and play a beep sound when drowsiness is detected.</li>
    <li>Reset the alert if the eyes are open again.</li>
</ol>

<h4>Issues and Troubleshooting</h4>
Low frame rate: Try reducing the resolution of the webcam capture or use a faster machine.<br>
Drowsiness not detected properly: Adjust the EAR threshold value in the code as per the testing scenario.<br>

<hr>
<h3>Author: Yaksh Shah</h3>
<b>GitHub: https://github.com/yaksh-shah2704</b>
