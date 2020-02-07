import os
import sys
import subprocess
import shutil


data_path = './KTH'

video_files=os.listdir(data_path + '/data/running/')
video_files.sort()


args=['ffmpeg', '-i']
for video in video_files:

	video_name = video[:-11]

	frame_name = '%d.jpg'
	os.makedirs(data_path + '/data/frames/running/'+video_name)
	args.append(data_path + '/data/running/'+video)
	args.append(data_path + '/data/frames/running/'+video_name+'/'+frame_name)
	ffmpeg_call = ' '.join(args)

	subprocess.call(ffmpeg_call, shell=True)
	args=['ffmpeg', '-i']
	if (video_files.index(video) + 1) % 50 == 0:
		print ('Completed till video : ', (video_files.index(video) + 1))

print ('[MESSAGE]	Frames extracted from all videos')
