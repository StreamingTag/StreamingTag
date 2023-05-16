# StreamingTag (MobiCom'22)

This is an official Github repository for the MobiCom paper "StreamingTag: A Scalable Piracy Tracking Solution for Mobile Streaming Services." This project is built upon FFmpeg, GNU Scientific Library (GSL), and consists of C/Python.
[[Project homepage]](https://streamingtag.github.io/) [[Paper]](https://dl.acm.org/doi/10.1145/3495243.3560521)

If you use our work for research, please cite it.
```
@inproceedings{streamingtag,
  title={StreamingTag: A Scalable Piracy Tracking Solution for Mobile Streaming Services},
  author={Jin, Xinqi and Dang, Fan and Fu, Qi-An and Li, Lingkun and Peng, Guanyan and Chen, Xinlei and Liu, Kebin and Liu, Yunhao},
  booktitle={Proceedings of the 28th Annual International Conference on Mobile Computing and Networking},
  pages={596--608},
  year={2022}
}
```

## Project structure
```
./StreamingTag                 # Python: implementing the slidingDTW algorithm & watermark extraction
./third_party
├── StreamingTag-FFmpeg                 # C: the customized FFmpeg with out watermarking scheme implemented as its filter
├── StreamingTag-gsl                 # C: the customized gsl supporting Discrete Wavelet Transfrom (DWT) with a specified level
```

## Prerequisites

* OS: Ubuntu (we test our implementation on Ubuntu 22.04)
* Python 3.9

## Guide
We show how to measure the robustness against screen recording-based NR attack in this guide.

### 1. Setup
* Install the required Python packages
```
cd ${PROJECT_DIR}/StreamingTag
pip install -r requirements.txt
```
* Follow the [guide](third_party/StreamingTag-gsl/INSTALL) to install the customized FFmpeg
* Follow the [guide](third_party/StreamingTag-FFmpeg/README.md) to install the customized FFmpeg

### 2. Embed a single bit into each segment of a video
* To embed a bit $0$:
```
ffmpeg -i ${original_video_path} -vf watermark=keyFrameInterval=${frame_rate_of_original_video}:numEmbeddedFrames=${numEmbeddedFramesPerSegment}:strength=${strength}:implicitSVD=1:size=512:mode=3:outputFn=embeddingPosition.txt ${watermarked_video_path}
```
* Alternatively, to embed a bit $1$:
```
ffmpeg -i ${original_video_path} -vf watermark=keyFrameInterval=${frame_rate_of_original_video}:numEmbeddedFrames=${numEmbeddedFramesPerSegment}:strength=${strength}:implicitSVD=1:size=512:mode=3:outputFn=embeddingPosition.txt ${watermarked_video_path}
```

Note that ${numEmbeddedFramesPerSegment} corresponds to $n_e$ in the paper, and ${strength} equals to $1\pm\alpha$ (where $\alpha$ denotes the watermark strength in the paper). 

### 3. Generate the pirated video

Copy the watermarked video to a phone, play it on the phone in the full-screen mode, and record it via some software-based screen recorder to generate the pirated copy. Then, copy the recorded video to the computer.

### 4. Pre-process the pirated video (Optional)
We mainly test the case where the frame rate of the pirated video equals to that of the watermarked video, so we recommend to keep the frame rate of the recorded video the same as that of the original video:
```
ffmpeg -i ${recorded_video_path} -vf fps=fps=${frame_rate_of_original_video} ${prepared_recorded_video_path}
rm ${recorded_video_path}
mv ${prepared_recorded_video_path} ${recorded_video_path}
```



### 5. Determine the value of `offset`
* First, convert the first 200 frames of the pirated/original video to images. Use a larger value for the parameter '-l' if necessary.
```
cd StreamingTag
python video-to-images.py -p ${original_video_folder} -v ${original_video_filename} -i image -l 200
python video-to-images.py -p ${recorded_video_folder} -v ${recorded_video_filename} -i image -l 200
```
* Second, manually check how many more frames the beginning of the pirated video has than the original video.


### 6. Conduct the measurement
```
  python main.py -o ${original_video_path} -r ${recorded_video_path} --offset ${offset} --owl 8 --rwl 8 -b ${embedded_bit} --ne ${numEmbeddedFramesPerSegment} -s 512 -f embeddingPosition.txt  
```
The slidingDTW algorithm is used in this measurement.