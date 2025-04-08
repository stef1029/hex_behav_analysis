import deeplabcut
import sys


def main(video_path, config_path):
    deeplabcut.create_labeled_video(config_path,
                                [video_path],
                                videotype='.avi',
                                draw_skeleton=True,
                                trailpoints=0,
                                displaycropped=False,
                                overwrite=True)

if __name__ == "__main__":

    video_path = sys.argv[1]

    config_path = r'/cephfs2/dwelch/6-choice_behaviour_DLC_model/config.yaml'

    main(video_path, config_path)