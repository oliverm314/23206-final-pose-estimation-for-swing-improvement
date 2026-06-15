import os
import argparse
import sys

def ensure_trailing_sep(path):
    return path if path.endswith(os.sep) else path + os.sep


def main():
    parser = argparse.ArgumentParser(description="Run golf swing analyzer on a single video and collect outputs into a folder.")
    parser.add_argument('video', help='Path to input video (mp4)')
    parser.add_argument('-o', '--out', help='Output folder (will be created)', default='output')
    args = parser.parse_args()

    video_path = args.video
    out_folder = args.out

    if not os.path.isfile(video_path):
        print(f"Input video not found: {video_path}")
        sys.exit(1)

    os.makedirs(out_folder, exist_ok=True)

    base = os.path.splitext(os.path.basename(video_path))[0]
    csv_path = os.path.join(out_folder, base + '.csv')
    annotated_video = os.path.join(out_folder, base + '_annotated.mp4')

    print('Step 1/3: Running MediaPipe pose estimation and creating CSV + annotated video...')
    try:
        from MediaPipe_class import MediaPipe_PoseEstimation
        mp = MediaPipe_PoseEstimation(video_path, csv_path, annotated_video)
        mp.process_video()
    except Exception as e:
        print('Error while running MediaPipe:', e)
        sys.exit(1)

    print('Step 2/3: Processing CSV, extracting key frames and running analysis...')
    try:
        from process_swing import DataProcessor, Evaluator, VideoProcessor

        folder_path = ensure_trailing_sep(out_folder)
        dp = DataProcessor(folder_path)
        dp.preprocess_data()

        ev = Evaluator(dp)
        vp = VideoProcessor(folder_path, dp, ev)

        vp.print_swing_analysis()
        vp.save_frame()
    except Exception as e:
        print('Error while analyzing swing:', e)
        sys.exit(1)

    print('Step 3/3: Done.')
    print('Outputs saved to:', os.path.abspath(out_folder))
    print(' - CSV:', os.path.abspath(csv_path))
    print(' - Annotated video:', os.path.abspath(annotated_video))
    print(' - Extracted frames: *_frame_address/top/contact.jpg in the output folder')


if __name__ == '__main__':
    main()
