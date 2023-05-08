# python3 watermark-extraction.py -o .\video1\bit0\out60\ -r .\video1\bit0\xiaomi\ -y .\video1\bit0\xiaomi-YUV\ -f video1/bit0/video1-bit0-xiaomi-sync_result.txt --rng_values_fn interval_25_ne_1_seed_0_nframes_25000_h_816_w_1920_size_512.txt -b 0 -s 512
# python3 watermark-extraction.py -o .\video3\bit1\out60\ -r .\video1\bit0\xiaomi\ -y .\video1\bit0\xiaomi-YUV\ -f video1/bit0/video1-bit0-xiaomi-sync_result.txt --rng_values_fn interval_25_ne_1_seed_0_nframes_25000_h_816_w_1920_size_512.txt -b 0 -s 512 -j .\video1\bit0\xiaomi.json
import argparse
import pywt
import os
import numpy as np
import json
import cv2
from sync import slidingDTW, VideoCapture

def readRecordedRegion(cap, frame_index, row_offset, col_offset, region_size):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    if ret:
        print(f"readRecordedRegion {frame_index} {row_offset} {col_offset} {region_size}")
        region = frame[row_offset:row_offset+region_size, col_offset:col_offset+region_size]
        if (region.shape[0] != region_size) or (region.shape[1] != region_size):
            print(f"frame.shape={frame.shape}")
            print(f"{row_offset}:{row_offset+region_size}:{col_offset}:{col_offset+region_size}")
            print(f"Warning: frame_index={frame_index} region.shape={region.shape} region_size={region_size}")
        region=region[:min(region.shape[:2]), :min(region.shape[:2]), :]
        print(region.shape)
        return region
    else:
        return None

def readRNGValues(fn):
    '''
    Format: #index #row_pos #col_pos
    '''
    result = []
    with open(fn, "r") as f:
        while True:
            line = f.readline()
            if not line:
                break
            line = line.strip().split(" ")
            assert len(line) == 3
            result.append([int(item) for item in line])
        f.close()
    #print(result)
    #exit(0)
    return result

def sumLH3SingularValues(region, original_region_size):
    #print(region)
    assert region.shape[0] == region.shape[1]
    shape = region.shape
    print(shape)
    print(cv2.cvtColor(region, cv2.COLOR_BGR2YUV_I420).shape)
    region = cv2.cvtColor(region, cv2.COLOR_BGR2YUV_I420)[:shape[0], :shape[1]]
    region = cv2.resize(region, (original_region_size, original_region_size), interpolation=cv2.INTER_AREA if original_region_size<=region.shape[0] else cv2.INTER_CUBIC)
    coeffs = pywt.wavedec2(region, 'haar', level=3)
    LH3 = coeffs[1][1]
    print(LH3.shape)
    #print(np.sum(np.abs(LH3)))
    singular_values = np.linalg.svd(LH3)[1]
    sum = np.sum(singular_values)
    return sum

def get_recorded_video_region(frame_index, row_offset, col_offset, originalShape, recordedShape, invResult):
    if frame_index in invResult.keys():
        recorded_frame_index = invResult[frame_index]
        originalHeight, originalWidth = originalShape
        screenHeight, screenWidth = recordedShape
        if screenWidth/screenHeight > originalWidth/originalHeight:
            recordedHeight = screenHeight
            recordedWidth = int(recordedHeight * originalWidth / originalHeight)
        else:
            recordedWidth = screenWidth
            recordedHeight = int(recordedWidth * originalHeight / originalWidth)
        recorded_row_offset = (screenHeight-recordedHeight)//2 + int(recordedHeight/originalHeight*row_offset)
        recorded_col_offset = (screenWidth-recordedWidth)//2 + int(recordedWidth/originalWidth*col_offset)
        return recorded_frame_index, int(recorded_row_offset), int(recorded_col_offset)
    else:
        return None, None, None


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    
    # sync
    parser.add_argument("-o", "--origin_video_path", required=True,)
    parser.add_argument("-r", "--recorded_video_path", required=True,)
    parser.add_argument("--offset", help="the number of skipped frames in the recorded video", \
        required=True, type=int)
    parser.add_argument("--owl", help="window length for original video capture", required=True, type=int)
    parser.add_argument("--rwl", help="window length for recorded video capture", required=True, type=int)

    # watermark extraction
    parser.add_argument("-b", "--bit", help="the correct bit", type=int, required=True)
    parser.add_argument("-n", "--ne", help="the number of embedding regions in each segment", type=int, required=True)
    parser.add_argument("-s", "--region_size", default=512, type=int, required=True)
    parser.add_argument("-f", "--file", help="the file containing watermark position information (including frame_index, row_offset, col_offset)", required=True)
    parser.add_argument("-j", "--json", help="the json file used to store slidingDTW results", default="None")

    ARGS = parser.parse_args()
    assert ARGS.owl>=2 and ARGS.rwl>=2
    bit = ARGS.bit
    region_size = ARGS.region_size

    origin_video_cap = VideoCapture(ARGS.origin_video_path, ARGS.owl)
    recorded_video_cap = VideoCapture(ARGS.recorded_video_path, ARGS.rwl, ARGS.offset)
    if not os.path.exists(ARGS.json):
        _, invResult = slidingDTW(recorded_video_cap, origin_video_cap, 1)
        with open(ARGS.json, "w") as f:
            f.write(json.dumps(invResult))
            f.flush()
    else:
        with open(ARGS.json, "r") as f:
            invResult = json.loads(f.readlines()[0])
            invResult = {int(k): v for k, v in invResult.items()}


    origin_video_cap = cv2.VideoCapture(ARGS.origin_video_path)
    recorded_video_cap = cv2.VideoCapture(ARGS.recorded_video_path)
    originalShape = (origin_video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT), origin_video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    recordedShape = (recorded_video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT), recorded_video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    
    originalHeight, originalWidth = originalShape
    screenHeight, screenWidth = recordedShape
    if screenWidth/screenHeight > originalWidth/originalHeight:
        recorded_region_size = int(screenHeight/originalHeight*region_size)
    else:
        recorded_region_size = int(screenWidth/originalWidth*region_size)

    num_matched = 0
    num_correct = 0
    num_mismatched = 0

    with open(ARGS.file, "r") as f:
        num_correct_single_segment = 0
        num_wrong_single_segment = 0
        positions = f.readlines()
        for line_index, line in enumerate(positions):
            line = line.strip()
            frame_index = int(line.split(" ")[0])
            row_offset = int(line.split(" ")[1])
            col_offset = int(line.split(" ")[2])
            print(f"line {frame_index} {row_offset} {col_offset}")
            recorded_frame_index, recorded_row_offset, recorded_col_offset = get_recorded_video_region(frame_index, row_offset, col_offset, originalShape, recordedShape, invResult)
            recorded_frame_index1, _, _ = get_recorded_video_region(frame_index-1, row_offset, col_offset, originalShape, recordedShape, invResult)
            if recorded_frame_index1 is not None:
                recorded_frame_index1 += ARGS.offset
            recorded_frame_index2, _, _ = get_recorded_video_region(frame_index+1, row_offset, col_offset, originalShape, recordedShape, invResult)
            if recorded_frame_index2 is not None:
                recorded_frame_index2 += ARGS.offset
            if (recorded_frame_index is not None) and not (recorded_frame_index1 is None and recorded_frame_index2 is None):
                originalRegion = readRecordedRegion(origin_video_cap, frame_index, row_offset, col_offset, region_size)
                referenceRegion1 = None
                if recorded_frame_index1 is not None:
                    referenceRegion1 = readRecordedRegion(recorded_video_cap, recorded_frame_index1, recorded_row_offset, recorded_col_offset, recorded_region_size)
                referenceRegion2 = None
                if recorded_frame_index2 is not None:
                    referenceRegion2 = readRecordedRegion(recorded_video_cap, recorded_frame_index2, recorded_row_offset, recorded_col_offset, recorded_region_size)
                embeddingRegion = readRecordedRegion(recorded_video_cap, recorded_frame_index+ARGS.offset, recorded_row_offset, recorded_col_offset, recorded_region_size)
                assert not ((referenceRegion1 is None) and (referenceRegion2 is None))
                if referenceRegion1 is None:
                    referenceRegion = referenceRegion2
                elif referenceRegion2 is None:
                    referenceRegion = referenceRegion1
                else:
                    if cv2.sumElems(cv2.absdiff(referenceRegion1, embeddingRegion)) < cv2.sumElems(cv2.absdiff(referenceRegion2, embeddingRegion)):
                        referenceRegion = referenceRegion1
                    else:
                        referenceRegion = referenceRegion2
                sum0 = sumLH3SingularValues(referenceRegion, region_size)
                sum1 = sumLH3SingularValues(embeddingRegion, region_size)
                cv2.imwrite(f"{frame_index}-original.png", originalRegion)
                cv2.imwrite(f"{frame_index}.png", embeddingRegion)
                cv2.imwrite(f"{frame_index}-reference.png", referenceRegion)
                print(f"ratio is {sum1} {sum0} {sum1/sum0}")
                if (sum1 > sum0 and bit == 0) or (sum1 < sum0 and bit == 1):
                    num_correct_single_segment += 1
                else:
                    num_wrong_single_segment += 1
            if (line_index+1) % ARGS.ne == 0:
                if num_correct_single_segment==num_wrong_single_segment:
                    num_mismatched += 1
                else:
                    num_matched += 1
                    if num_correct_single_segment > num_wrong_single_segment:
                        num_correct += 1
                num_correct_single_segment = 0
                num_wrong_single_segment = 0
                print("Info: Accuracy is {}/{}".format(num_correct, num_matched))
        f.close()

    print("Info: completed. Overall accuracy is {}/{}. {} frames mismatched.".format(num_correct, num_matched, num_mismatched))
