# import subprocess
from fcutils.video.utils import (
    get_cap_from_images_folder,
    save_videocap_to_video,
)
import click
import logging


def animate_from_images(folder, savepath, fps):
    cap = get_cap_from_images_folder(folder, img_format="%2d.png")
    save_videocap_to_video(cap, savepath, ".mp4", fps=fps)

    gifpath = savepath.replace(".mp4", ".gif")
    logging.info(
        "To save the video as GIF, use: \n"
        + f'ffmpeg -i "{savepath}" -f gif "{gifpath}"'
    )
    # subprocess.call(["ffmpeg", "-i", f"{savepath}", "-f", "gif", f"{gifpath}"])
    # print(f"\n\n\n saved gif at: {gifpath}")


@click.command()
@click.argument("folder")
@click.argument("savepath")
def main(folder, savepath):
    animate_from_images(folder, savepath)
