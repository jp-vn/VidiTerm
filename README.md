# terminal vid player rendered w/ unicode braille

###### i love unicode braille

C++ rewrite https://www.youtube.com/watch?v=R4U5uTmzjb4

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

## due time has passed and this was solved by rewriting in cpp
