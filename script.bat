REM Activate the conda environment
CALL conda activate zhiqi
REM Change to the directory where your Python scripts are located
cd C:\Science\private_vision
REM Run python scripts
python cifar_DP.py --lr 0.002 --epochs 3 --model vit_base_patch16_224
PAUSE