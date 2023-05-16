import argparse
import re
import cv2
import os
import copy
import numpy as np
from enum import Enum, unique
from fastdtw import dtw
    
last_epoch = False

@unique
class FrameState(Enum):
    unchecked = 1
    matched = 2
    dropped = 3
    inserted = 4
    skipped = 5

def cropRecordedFrame(originalFrame, recordedFrame):
    originalHeight, originalWidth, _ = originalFrame.shape
    screenHeight, screenWidth, _ = recordedFrame.shape
    if screenWidth/screenHeight > originalWidth/originalHeight:
        recordedHeight = screenHeight
        recordedWidth = int(recordedHeight * originalWidth / originalHeight)
    else:
        recordedWidth = screenWidth
        recordedHeight = int(recordedWidth * originalHeight / originalWidth)
    #print(f"originalHeight = {originalHeight} originalWidth = {originalWidth} screenHeight = {screenHeight} screenWidth = {screenWidth} recordedHeight = {recordedHeight} recordedWidth = {recordedWidth}")
    recordedFrame = cv2.resize(recordedFrame[(screenHeight-recordedHeight)//2:(screenHeight-recordedHeight)//2+recordedHeight, (screenWidth-recordedWidth)//2:(screenWidth-recordedWidth)//2+recordedWidth], (originalWidth, originalHeight), interpolation = cv2.INTER_LINEAR)
    return recordedFrame

def diff(originalFrame, recordedFrame):
    recordedFrame = cropRecordedFrame(originalFrame, recordedFrame)
    return np.sum(cv2.absdiff(recordedFrame, originalFrame))
   
class Frame:
    def __init__(self, index, frame):
        self.index = index
        self.frame = frame.astype(np.float32)
        self.state = FrameState.unchecked
        self.matchedIndex = -1
        self.dist = -1 # -1 means uninitialized

class VideoCapture:
    def __init__(self, videoFn, windowLength, offset=0):
        global last_epoch
        self.videoFn = videoFn
        self.cap = cv2.VideoCapture(self.videoFn)
        self.offset = offset
        self.numFrames = 0
        self.windowLength = windowLength
        self.framesWindow = []
        for i in range(offset):
            self.cap.read()
        for i in range(self.windowLength):
            ret, frame = self.cap.read()
            if ret:
                self.numFrames += 1
                self.framesWindow.append(Frame(i,frame))
            else:
                last_epoch = True
                break
    
    def update(self, remove_first=True):
        global last_epoch
        idx = self.numFrames
        if remove_first:
            self.framesWindow = self.framesWindow[1:]
        ret, frame = self.cap.read()
        if ret:
            self.numFrames += 1
            self.framesWindow.append(Frame(idx, frame))
            return True
        else:
            print(f"End of video {self.videoFn}\nTotal number of frames: {self.numFrames}")
            last_epoch = True
            return False

    def removeOneCheckedFrame(self):
        if len(self.framesWindow) == 0:
            return False, None
        if self.framesWindow[0].state == FrameState.unchecked:
            return False, None
        frame = self.framesWindow[0]
        self.framesWindow = self.framesWindow[1:]
        return True, frame

def isMismatch(original_frame, recorded_frame, candidate_original_frames, candidate_recorded_frames):
    diffVal = diff(original_frame, recorded_frame)
    for candidate in candidate_original_frames:
        if diff(candidate, recorded_frame) < diffVal:
            return True
    for candidate in candidate_recorded_frames:
        if diff(original_frame, candidate) < diffVal:
            return True
    return False

def slidingDTW(RecordedVideoCapture, OriginalVideoCapture, isMismatchHalfWindowLength):
    global last_epoch
    epoch = 0
    result = {}
    invResult = {}
    debug_log = open("debug.log", "w")
    def setMatchedStateForOriginalFrame(original_frame, recorded_frame):
        print(f"setMatchedStateForOriginalFrame: frame_index={original_frame.index}")
        if original_frame.state != FrameState.matched:
            original_frame.state = FrameState.matched
            original_frame.matchedIndex = recorded_frame.index
            original_frame.dist = diff(original_frame.frame, recorded_frame.frame)
        else:
            dist = diff(original_frame.frame, recorded_frame.frame)
            if dist < original_frame.dist:
                original_frame.dist = dist
                original_frame.matchedIndex = recorded_frame.index

    

    while True:
        epoch += 1
        lastRecordedWindow = []
        lastOriginalWindow = []
        # slide to the next window:
        while True:
            ret = RecordedVideoCapture.removeOneCheckedFrame()
            if ret[0]:
                lastRecordedWindow.append(ret[1].frame)
            else:
                break
        while len(RecordedVideoCapture.framesWindow) <= RecordedVideoCapture.windowLength:
            if not RecordedVideoCapture.update(remove_first=False):
                break
        while True:
            ret = OriginalVideoCapture.removeOneCheckedFrame()
            if ret[0]:
                lastOriginalWindow.append(ret[1].frame)
            else:
                break
        while len(OriginalVideoCapture.framesWindow) <= OriginalVideoCapture.windowLength:
            if not OriginalVideoCapture.update(remove_first=False):
                break
        # match
        if len(OriginalVideoCapture.framesWindow) == 0 or len(RecordedVideoCapture.framesWindow) == 0:
            assert last_epoch
            pass
        else:
            # run dtw:
            original_frames = []
            for original_frame in OriginalVideoCapture.framesWindow[:OriginalVideoCapture.windowLength]:
                original_frames.append(np.expand_dims(original_frame.frame, 0))
                # debug
                # cv2.imwrite(f"{original_frame.index}.png", original_frame.frame)
            recorded_frames = []
            for recorded_frame in RecordedVideoCapture.framesWindow[:RecordedVideoCapture.windowLength]:
                recorded_frame = cropRecordedFrame(original_frames[0][0], recorded_frame.frame)
                recorded_frames.append(np.expand_dims(recorded_frame, 0))
                # debug
                # index = recorded_frame.index+RecordedVideoCapture.offset
                # cv2.imwrite(f"{index}.png", recorded_frame)
            original_frames = np.concatenate(original_frames, axis=0)
            recorded_frames = np.concatenate(recorded_frames, axis=0)
            #print(original_frames.shape)
            #print(recorded_frames.shape)
            distance, path = dtw(original_frames, recorded_frames, dist=diff)
            # pre-fetch for judging mismatch
            for _ in range(isMismatchHalfWindowLength):
                OriginalVideoCapture.update(remove_first=False)
                RecordedVideoCapture.update(remove_first=False)
            mismatchedOriginalFramesIndices = []
            mismatchedRecordedFramesIndices = []
            for pair in path:
                # Key insight: we need to remove mismatched pairs (which could be caused by dropped frames or duplicated frames)
                if not isMismatch(original_frames[pair[0]], recorded_frames[pair[1]], (lastOriginalWindow[pair[0]-isMismatchHalfWindowLength:] if pair[0]<isMismatchHalfWindowLength else []) + [ele for ele in original_frames[max(0, pair[0]-isMismatchHalfWindowLength):pair[0]]]+[ele for ele in original_frames[pair[0]+1:pair[0]+1+isMismatchHalfWindowLength]], (lastRecordedWindow[pair[1]-isMismatchHalfWindowLength:] if pair[1]<isMismatchHalfWindowLength else []) + [ele for ele in recorded_frames[max(0, pair[1]-isMismatchHalfWindowLength):pair[1]]]+[ele for ele in recorded_frames[pair[1]+1:pair[1]+1+isMismatchHalfWindowLength]]):
                    # Key insight: match it with the most similar frame in the recorded video (since frames may be duplicated in the recorded video):
                    setMatchedStateForOriginalFrame(OriginalVideoCapture.framesWindow[pair[0]], RecordedVideoCapture.framesWindow[pair[1]])
                    # Multiple frames in the recorded video can match to the same frame in the original video; but not vice versa
                    RecordedVideoCapture.framesWindow[pair[1]].state = FrameState.matched
                    RecordedVideoCapture.framesWindow[pair[1]].matchedIndex = OriginalVideoCapture.framesWindow[pair[0]].index
                else:
                    print(f"Mismatch between original frame ({OriginalVideoCapture.framesWindow[pair[0]].index+OriginalVideoCapture.offset}.png) and recorded frame ({RecordedVideoCapture.framesWindow[pair[1]].index+RecordedVideoCapture.offset}.png)")
                    debug_log.write(f"Mismatch: original/{OriginalVideoCapture.framesWindow[pair[0]].index+OriginalVideoCapture.offset}.png recorded/{RecordedVideoCapture.framesWindow[pair[1]].index+RecordedVideoCapture.offset}.png\n")
                    debug_log.flush()
            for idx, frame in enumerate(OriginalVideoCapture.framesWindow):
                if frame.state != FrameState.matched:
                    mismatchedOriginalFramesIndices.append(idx)
            for idx, frame in enumerate(RecordedVideoCapture.framesWindow):
                if frame.state != FrameState.matched:
                    mismatchedRecordedFramesIndices.append(idx)
            startIdx = len(OriginalVideoCapture.framesWindow) - 1
            while startIdx in mismatchedOriginalFramesIndices:
                startIdx -= 1
            for idx in mismatchedOriginalFramesIndices:
                if idx >= startIdx + 1:
                    # Key insight: if some frames from the original video are duplicated in the recorded video, there could exist mismatched frames located at the end of the original video's sliding window and corresponding to frames from the next window of the recorded video
                    OriginalVideoCapture.framesWindow[idx].state = FrameState.unchecked
                else:
                    OriginalVideoCapture.framesWindow[idx].state = FrameState.dropped
            if startIdx == -1:
                OriginalVideoCapture.framesWindow[0].state = FrameState.skipped
            startIdx = len(RecordedVideoCapture.framesWindow) - 1
            while startIdx in mismatchedRecordedFramesIndices:
                startIdx -= 1
            for idx in mismatchedRecordedFramesIndices:
                if idx >= startIdx + 1:
                    # Key insight: if some frames from the original video are dropped in the recorded video, there could exist mismatched frames located at the end of the recorded video's sliding window and corresponding to frames from the next window of the original video
                    RecordedVideoCapture.framesWindow[idx].state = FrameState.unchecked
                else:
                    RecordedVideoCapture.framesWindow[idx].state = FrameState.inserted
            if startIdx == -1:
                RecordedVideoCapture.framesWindow[0].state = FrameState.skipped
        # print result
        print(f"[Epoch {epoch}]")
        for frame in OriginalVideoCapture.framesWindow:
            if frame.state == FrameState.matched:
                print(f"frame index in {RecordedVideoCapture.videoFn}: {frame.matchedIndex} ({frame.matchedIndex+RecordedVideoCapture.offset}.png) matched index in {OriginalVideoCapture.videoFn}: {frame.index} ({frame.index+OriginalVideoCapture.offset}.png)")
                result[frame.matchedIndex] = frame.index
                invResult[frame.index] = frame.matchedIndex
                debug_log.write(f"Match: original/{frame.index+OriginalVideoCapture.offset}.png recorded/{frame.matchedIndex+RecordedVideoCapture.offset}.png\n")
                debug_log.flush()
        if last_epoch:
            return result, invResult