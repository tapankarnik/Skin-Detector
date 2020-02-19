# Skin Detector

A modified implementation of a paper titled, 'Skin detection and lightweight encryption for privacy protection in realtime surveillance applications'. Except the encryption.
Tested on Python 3.5.2. Should work on later versions.
Install all the requirements through the requirements file using

pip install -r requirements.txt

###
(Optional but highly recommended)
Create a virtual environment using

virtualenv venv
source venv/bin/activate
###

Run the file using the following
python skindetector.py --input abc.jpg
OR
python skindetector.py --input abc.jpg --width width_in_pixels
Where width_in_pixels is the width you want to resize the image to.




