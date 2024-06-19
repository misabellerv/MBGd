import os

from utils.vid_utils import save_frames

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(description='Save videos frames.')

    parser.add_argument('--datapath',
                        type=str,
                        default='../data/_under_construction',
                        help='Data path of videos to extract frames from.')

    parser.add_argument('--save_every', type=int, default=24, help='Number of frames to skip.')

    args = parser.parse_args()

    data_path = args.datapath
    save_every = args.save_every

    for (dirpath, dirnames, filenames) in os.walk(data_path):

        if len(filenames) == 0:
            continue

        for filename in filenames:

            if filename.lower().endswith(('.mov', '.mp4', '.avi')):

                input_path = os.path.join(dirpath, filename)
                output_path = os.path.join(dirpath, os.path.splitext(filename)[0], 'frames')

                print(input_path)
                print(output_path)
                save_frames(input_path, output_path, save_every=save_every)