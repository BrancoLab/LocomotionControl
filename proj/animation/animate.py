
import os
import subprocess
from fcutils.video.utils import get_cap_from_images_folder, save_videocap_to_video

def animate_from_images(folder, savepath):    
    print('Loading images')
    cap = get_cap_from_images_folder(folder, img_format="%2d.png")

    print('Saving to video')
    save_videocap_to_video(cap, savepath, '.mp4')

    gifpath = savepath.replace('.mp4', '.gif')
    print('To save the video as GIF, use: \n'+
            f'ffmpeg -i "{savepath}" -f gif "{gifpath}"'
    )
    subprocess.call(['ffmpeg', '-i', f'{savepath}', '-f',  'gif', f'{gifpath}'])

    print(f'\n\n\n saved gif at: {gifpath}')