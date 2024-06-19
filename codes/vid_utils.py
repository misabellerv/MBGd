import fnmatch
import os

import imageio
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.autonotebook import tqdm

import cv2
from utils.img_utils import compute_mse, compute_psnr, compute_ssim

sns.set('paper',
        'white',
        'colorblind',
        font_scale=2.2,
        rc={
            'lines.linewidth': 2,
            'figure.figsize': (10.0, 6.0),
            'image.interpolation': 'nearest',
            'image.cmap': 'gray'
        })


def generate_video_opencv(input_path,
                          output_path,
                          first_frame=0,
                          last_frame=None,
                          skip_frames=0,
                          codec='X264',
                          fps=None,
                          verbose=False):
    """Generates a video with the frames extracted from the input video using opencv.

    Arguments:
        input_path {str} -- input video path
        output_path {str} -- output video path

    Keyword Arguments:
        first_frame {int} -- first frame to consider (default: {0})
        last_frame {inte} -- last frame to consider. If None, consider last frame of original video (default: {None})
        skip_frames {int} -- Number of frames to skip (default: {0})
        codec {str} -- codec to generate the video (default: {'X264'})
        fps {float} -- video frame rate. If None, use the input video frame rate (default: {None})
        verbose {bool} -- Flag to output some information (default: {False})
    """

    print('This may take a while...')

    #     cam_matrix, profile = create_matrix_profile(FC, CC, KC)

    # Load video
    # video_input = videoObj(input_path)
    video_input = cv2.VideoCapture(input_path)

    # if codec is None:
    #     a, b, c, d = video_input.get(cv2.CAP_PROP_FOURCC)
    # codec = video_input.videoInfo.getCodecType()

    fourcc = cv2.VideoWriter.fourcc(*list(codec))

    # number of frames of video
    # nb_frames = video_input.videoInfo.getNumberOfFrames()
    nb_frames = video_input.get(cv2.CAP_PROP_FRAME_COUNT)

    if fps is None:
        # fps = video_input.videoInfo.getFrameRateFloat()
        fps = video_input.get(cv2.CAP_PROP_FPS)

    size = (int(video_input.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(video_input.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    # size = (video_input.videoInfo.getWidth(), video_input.videoInfo.getHeight())

    writer = cv2.VideoWriter(output_path, fourcc, fps, size)

    # if last_frame is not specified, use the total number of frames
    if last_frame is None:
        last_frame = int(nb_frames)

    print('generating video: {}'.format(output_path))
    for idx in tqdm(range(0, last_frame)):
        if verbose:
            tqdm.write('On frame {} of {}.'.format(idx, last_frame))

        # _, frame, _ = video_input.get_frame(idx)
        video_input.grab()
        frame = video_input.retrieve()[1]
        writer.write(frame)

    video_input.release()
    writer.release()


def generate_video_imageio(input_path, output_path, quality):
    """Generates a video with the frames extracted from the input video using imageio, ffmpeg wrapper.

    Arguments:
        input_path {str} -- input video path
        output_path {str} -- output video path
        quality {float} -- video quality. Based on CRF from ffmpeg. The higher the quality, the better the video but the larger the size
    """

    # Carrego o vídeo de origem e seu fps
    reader = imageio.get_reader(input_path)
    fps = reader.get_meta_data()['fps']
    # nb_frames = reader.get_meta_data()['nframes']

    # Crio o writer para gerar um vídeo de saída com qualidade 10 (menor compressão possível)
    writer = imageio.get_writer(output_path, fps=fps, quality=quality)

    print('generating video: {}'.format(output_path))
    for im in tqdm(reader):
        writer.append_data(im)

    writer.close()


def compute_video_size(filepath):
    """Returns the file size in MB

    Arguments:
        filepath {str} -- path to file

    Returns:
        float -- the file size in MB
    """

    return os.path.getsize(filepath) / (1024**2)


def compare_videos(filepath1,
                   filepath2,
                   first_frame=0,
                   last_frame=None,
                   compute_every=1,
                   verbose=False,
                   save_frames=False,
                   save_path=None,
                   debug=False):

    print('Comparing videos... This may take a while... \n')

    vid1 = cv2.VideoCapture(filepath1)
    vid2 = cv2.VideoCapture(filepath2)

    # # print all videos information
    print('Video 1')
    print_video_info(filepath1)
    print('Video 2')
    print_video_info(filepath2)

    # number of frames of both videos
    nb_frames_vid1 = vid1.get(cv2.CAP_PROP_FRAME_COUNT)
    nb_frames_vid2 = vid2.get(cv2.CAP_PROP_FRAME_COUNT)

    if nb_frames_vid1 != nb_frames_vid2:
        raise IOError('Videos have different number of frames!')

    # Initializing MSE_sum and SSIM_sum to compute mean
    mse = []
    psnr = []
    ssim = []

    # if last_frame is not specified, use the total number of frames
    if last_frame is None:
        last_frame = int(nb_frames_vid1)

    # count = 0

    for idx in tqdm(range(first_frame, last_frame)):

        # count += 1

        if verbose:
            tqdm.write('On frame {} of {}'.format(idx + 1, last_frame))

        vid1.grab()
        vid2.grab()

        if (idx % compute_every) == 0:

            frame_vid1 = vid1.retrieve()[1]
            frame_vid2 = vid2.retrieve()[1]

            if save_frames:

                if not os.path.exists(save_path):
                    os.makedirs(save_path)

                compression_level = 3
                ext_params = [cv2.IMWRITE_PNG_COMPRESSION, compression_level]

                cv2.imwrite(os.path.join(save_path, 'frame_{:05d}_original.png'.format(idx)),
                            frame_vid1, ext_params)
                cv2.imwrite(os.path.join(save_path, 'frame_{:05d}_compressed.png'.format(idx)),
                            frame_vid2, ext_params)

            # _, frame_vid1, _ = vid1.get_frame(idx)
            # _, frame_vid2, _ = vid2.get_frame(idx)

            if debug:
                # Draw and display the corners
                plt.subplot(1, 2, 1)
                plt.imshow(cv2.cvtColor(frame_vid1, cv2.COLOR_BGR2RGB))
                plt.axis('off')
                plt.subplot(1, 2, 2)
                plt.imshow(cv2.cvtColor(frame_vid2, cv2.COLOR_BGR2RGB))
                plt.axis('off')
                plt.show()

            mse.append(compute_mse(frame_vid1, frame_vid2))
            psnr.append(compute_psnr(frame_vid1, frame_vid2))
            ssim.append(compute_ssim(frame_vid1, frame_vid2))

    vid1.release()
    vid2.release()

    return mse, psnr, ssim


def print_video_info(filepath):

    filesize = compute_video_size(filepath)

    video = cv2.VideoCapture(filepath)

    fps = video.get(cv2.CAP_PROP_FPS)

    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    resolution = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)),
                  int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    print('{}'.format(filepath))
    print('video size: {:.2f} MB'.format(filesize))
    print('fps: {:.2f}'.format(fps))
    print('frames: {:d}'.format(total_frames))
    print('resolution: {}'.format(resolution))

    print('\n')


def save_frames(input_path,
                output_path,
                first_frame=0,
                last_frame=None,
                save_every=1,
                verbose=False):

    # Load video
    vid = cv2.VideoCapture(input_path)

    nb_frames_vid = vid.get(cv2.CAP_PROP_FRAME_COUNT)

    compression_level = 3
    ext_params = [cv2.IMWRITE_PNG_COMPRESSION, compression_level]

    if last_frame is None:
        last_frame = int(nb_frames_vid)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for idx in tqdm(range(first_frame, last_frame)):

        if verbose:
            tqdm.write('On frame {} of {}'.format(idx + 1, last_frame))

        vid.grab()

        if (idx % save_every) == 0:

            frame_vid = vid.retrieve()[1]

            cv2.imwrite(os.path.join(output_path, 'frame_{:05d}.png'.format(idx)), frame_vid,
                        ext_params)

    vid.release()


def find_annot_file(video_path, annot_folder):
    """Find annotation file based on video name.

    Arguments:
        video_path {str} -- video path
        annot_folder {str} -- folder where to look in order to find the annotation file

    Returns:
        str -- [The annotation file path]
    """

    vid_filename = os.path.split(video_path)[-1]
    vid_filename, vid_ext = os.path.splitext(vid_filename)

    found = False

    for (dirpath, dirnames, filenames) in os.walk(annot_folder):

        if len(filenames) == 0:
            continue

        for file_name in filenames:

            if fnmatch.fnmatch(file_name, vid_filename + '.txt'):
                annot_path = os.path.join(dirpath, file_name)
                found = True
                break
        if found:
            break
    return annot_path
