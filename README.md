# in terminal vid player in unicode braille

###### i love unicode braille

![preview](https://github-production-user-asset-6210df.s3.amazonaws.com/180278020/418462265-ef0c0eb6-49c8-4305-8f94-e529344933c4.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVCODYLSA53PQK4ZA%2F20250303%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20250303T073855Z&X-Amz-Expires=300&X-Amz-Signature=9a87574a320702ca0b165de420aebbe589024a8e26681b36d1a39e9dab26abad&X-Amz-SignedHeaders=host)

### Youtube Demo [https://www.youtube.com/watch?v=86BE5R7Ux6M]

[![YouTube](http://i.ytimg.com/vi/86BE5R7Ux6M/hqdefault.jpg)](https://www.youtube.com/watch?v=86BE5R7Ux6M)

### reqs

     - Terminal w/ unicode and ANSI color support
     - Python 3.6+
     - opencv-python
     - numpy
     - FFmpeg (need ffplay for audio playback)

    install ffmpeg (if using brew as pckg manager -> brew install ffmpeg)
    pip3 install -r requirements.txt
    cd to the folder where the python script is
    python3 terminal_mp4_player.py /path/to/your/video.mp4

    ctrl + c to exit the player

still buggy

tested in iterm2, runs smoother in kitty but the aspect ratio was being funky

frames not consistent across different videos i tested resulting in audio video desync for some vids

etc

will get back to it in due time
