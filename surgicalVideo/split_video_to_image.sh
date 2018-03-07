mkdir video01
./ffmpeg_static/ffmpeg -i video01.mp4 -r 1 -q:v 2 -f image2 video01/video01-%d.jpg