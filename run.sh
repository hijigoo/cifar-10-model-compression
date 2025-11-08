#!/bin/bash

# CIFAR-10 Model Compression Project
# This script runs the complete experiment pipeline

set -e  # Exit on error

echo "======================================"
echo "Model Compression Midterm Project"
echo "CIFAR-10 Classification with Pruning"
echo "======================================"

# Configuration
SEEDS=(42 123 456)
METHODS=("magnitude" "structured" "lottery_ticket")
SPARSITY_LEVELS=(0.3 0.5 0.7 0.9 0.95)
DENSE_EPOCHS=200
FINETUNE_EPOCHS=100

# Parse command line arguments
QUICK_MODE=false
SKIP_DENSE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)
            QUICK_MODE=true
            shift
            ;;
        --skip-dense)
            SKIP_DENSE=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: ./run.sh [--quick] [--skip-dense]"
            echo "  --quick: Run with reduced epochs for testing"
            echo "  --skip-dense: Skip dense model training"
            exit 1
            ;;
    esac
done

# Adjust parameters for quick mode
if [ "$QUICK_MODE" = true ]; then
    echo "Running in QUICK MODE (reduced epochs)"
    DENSE_EPOCHS=10
    FINETUNE_EPOCHS=10
    SEEDS=(42)
    SPARSITY_LEVELS=(0.5 0.9)
fi

# Step 1: Setup environment
echo ""
echo "[1/5] Setting up environment..."
echo "--------------------------------------"

if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

echo "Activating virtual environment..."
source .venv/bin/activate

echo "Installing dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt

echo "✓ Environment ready"

# Step 2: Train dense baseline models
if [ "$SKIP_DENSE" = false ]; then
    echo ""
    echo "[2/5] Training dense baseline models..."
    echo "--------------------------------------"
    
    for seed in "${SEEDS[@]}"; do
        echo "Training dense model with seed=$seed..."
        python train_dense.py \
            --seed $seed \
            --epochs $DENSE_EPOCHS \
            --batch-size 128 \
            --lr 0.1
        
        echo "✓ Dense model (seed=$seed) completed"
        echo ""
    done
    
    echo "✓ All dense models trained"
else
    echo ""
    echo "[2/5] Skipping dense model training..."
    echo "--------------------------------------"
fi

# Step 3: Apply pruning and fine-tune
echo ""
echo "[3/5] Applying pruning methods and fine-tuning..."
echo "--------------------------------------"

total_experiments=$((${#METHODS[@]} * ${#SPARSITY_LEVELS[@]} * ${#SEEDS[@]}))
current=0

for method in "${METHODS[@]}"; do
    for sparsity in "${SPARSITY_LEVELS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            current=$((current + 1))
            echo ""
            echo "[$current/$total_experiments] Method: $method, Sparsity: $sparsity, Seed: $seed"
            
            python train_pruned.py \
                --method $method \
                --sparsity $sparsity \
                --seed $seed \
                --finetune-epochs $FINETUNE_EPOCHS \
                --finetune-lr 0.01 \
                --batch-size 128
            
            echo "✓ Completed"
        done
    done
done

echo ""
echo "✓ All pruning experiments completed"

# Step 4: Evaluate all models
echo ""
echo "[4/5] Evaluating all models..."
echo "--------------------------------------"

python evaluate_all.py

echo "✓ Evaluation completed"

# Step 5: Generate plots and tables
echo ""
echo "[5/5] Generating plots and tables..."
echo "--------------------------------------"

python plot_results.py

echo "✓ Plots and tables generated"

# Summary
echo ""
echo "======================================"
echo "All experiments completed successfully!"
echo "======================================"
echo ""
echo "Results location:"
echo "  - Checkpoints: experiments/checkpoints/"
echo "  - Results JSON: experiments/results/all_results.json"
echo "  - Plots: experiments/results/*.png"
echo "  - Tables: experiments/results/*.md, *.tex, *.csv"
echo ""
echo "Key outputs for report:"
echo "  1. experiments/results/tradeoff_curve.png"
echo "  2. experiments/results/efficiency_table.md"
echo "  3. experiments/results/all_results.json"
echo ""
echo "======================================"
