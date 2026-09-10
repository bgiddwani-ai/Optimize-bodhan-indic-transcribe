cp /data/indic-transcribe-flex-trtllm-engine/encoder/encoder.plan /data/models/riva-trt-canary-1b-multi-asr-offline-am-streaming-offline/1/engine/encoder/
cp canary.py /data/models/riva-trt-canary-1b-multi-asr-offline-am-streaming-offline/1/
CONFIG=/data/models/riva-trt-canary-1b-multi-asr-offline-am-streaming-offline/config.pbtxt
sed -E -i 's/preferred_batch_size: .*/preferred_batch_size: [32, 64]/; s/max_queue_delay_microseconds: [0-9]+/max_queue_delay_microseconds: 1000/' "$CONFIG"
