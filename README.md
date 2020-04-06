# Skin Detector

A modified implementation of a paper titled, 'Skin detection and lightweight encryption for privacy protection in realtime surveillance applications'. Except the encryption.
Tested on Python 3.5.2. Should work on later versions.

Install all the requirements through the requirements file using

<code>
pip install -r requirements.txt
</code>

### (Optional but highly recommended)

Create a virtual environment using

<code>
virtualenv venv

source venv/bin/activate
</code>

### Run the file using the following

<code>
python skindetector.py --input abc.jpg
</code>

OR

<code>
python skindetector.py --input abc.jpg --width width_in_pixels
</code>

Where width_in_pixels is the width you want to resize the image to.

