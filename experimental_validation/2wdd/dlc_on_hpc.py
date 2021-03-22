from deeplabcut import analyze_videos, filterpredictions

import argparse


def track(config_file, video, dest_fld):
    analyze_videos(
        config_file,
        [video],
        gputouse=0,
        destfolder=dest_fld,
        videotype=".avi",
        save_as_csv=False,
        dynamic=(False, 0.5, 10),
    )

    filterpredictions(config_file, [video], filtertype="median")


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        dest="config_file", type=str, help="path to config",
    )
    parser.add_argument(
        dest="video", type=str, help="path video file",
    )
    parser.add_argument(
        dest="dest_fld", type=str, help="path to dest fld",
    )
    return parser


def main():
    args = get_parser().parse_args()
    track(
        args.config_file, args.video, args.dest_fld,
    )


if __name__ == "__main__":
    main()
