srun  --mem=25GB --gres=gpu:01 --exclude=tikgpu[06-10] --pty bash -i
cd /itet-stor/sbuehrer/net_scratch/difflut
singularity exec --nv \
    --bind .:/workspace \
    --bind data:/workspace/data \
    --env PYTHONPATH=/workspace \
    containers/pytorch_universal_minimal.sif \
    python /workspace/experiments/mnist_handwritten_digits.py --dataset-fraction 1.0 --epochs 20




python mnist_handwritten_digits.py --dataset-fraction 0.1 --epochs 3


python experiments/visualize_results.py --results-dir /itet-stor/sbuehrer/net_scratch/difflut/results --output-dir /itet-stor/sbuehrer/net_scratch/difflut/results