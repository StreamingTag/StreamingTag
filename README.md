## Environment configuration

Use python 3.9 and install dependencies as follows:
```
pip install -r requirements.txt
```

---

## Measuring StreamingTag's security under the NR attack

### Step 1: Pre-processing the pirated video video

```
ffmpeg -i [recorded_video_filename] -vf fps=fps=[frame_rate_of_original_video] [prepared_recorded_video_filename]
```

### Step 2: Determining the value of `offset`

```
mkdir -p original
ffmpeg -i [origin_video_filename] original/%08d.png
mkdir -p prepared_recorded
ffmpeg -i [prepared_recorded_video_filename] prepared_recorded/%08d.png
```

Manually check how many more frames the beginning of the video [prepared_recorded_video_filename] has than the original video [origin_video_filename]. The number of extra frames can be set to [offset]. The above two calls of `ffmpeg` can be terminated early to save time and storage space, as long as the output of the calls is enough for you to find out the appropriate value of [offset].


### Step 3: Conducting the measurement

Optional: applying modification
```
ffmpeg -i [prepared_recorded_video_filename] -vf scale=w=0.75*iw:h=0.75*ih [output_filename]
ffmpeg -i [prepared_recorded_video_filename] -vf scale=w=1.25*iw:h=1.25*ih [output_filename]
ffmpeg -i [prepared_recorded_video_filename] -vf noise=alls=10*allf=0 noise/[output_filename]
ffmpeg -i [prepared_recorded_video_filename] -vf median=radius=7:radiusV=7 mf15/[output_filename]
```

```
python sync.py -o [origin_video_filename] -r [prepared_recorded_video_filename] --offset [offset]
```