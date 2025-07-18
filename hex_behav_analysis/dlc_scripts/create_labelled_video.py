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

    config_path = r'/cephfs2/srogers/2503_DLC_model_videos/LMDC_model_videos/LMDC-StefanRC-2025-03-11/config.yaml'

    main(video_path, config_path)