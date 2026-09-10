set -euo pipefail

python3 -m pip install nvidia-pyindex
python3 -m pip install --extra-index-url https://pypi.nvidia.com nvidia-eff
cd /home/CanaryNIM-Bharat-Final/nemo2riva
python3 -m pip install --no-deps .
export PYTHONPATH=/home/CanaryNIM-Bharat-Final/nemo2riva:$PYTHONPATH
nemo2riva --key=bodhan --out /data/indic-transcribe-flex.riva /data/indic-transcribe-flex.nemo

cd /home/CanaryNIM-Bharat-Final
python3 convert_checkpoint.py \
	--dtype=bfloat16 \
	--model_name "nvidia/canary-1b" \
        --model_path /data/indic-transcribe-flex.nemo  \
        --output_dir /data/indic-transcribe-flex-trtllm-checkpoint \
	/data/indic-transcribe-flex-trtllm-engine
