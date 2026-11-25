#!/bin/bash
# Stage A training on ONE node (4 A100) - starts sooner than the 2-node job
# when 2-node jobs wait in the queue; about half the steps per second.
# Same script as stage_a.sbatch (sourced below; its #SBATCH lines do not apply).
# To train a variant of a 2-node run with the same samples per optimizer step
# (4 GPUs x batch 8 x 2 accumulated batches = 8 GPUs x batch 8):
#   ABLATION_OF=<ref step.pt> ABLATION_ALLOW="<keys> train.grad_accum" \
#   EXTRA="--same-global-batch --set train.grad_accum=2 --set <keys>=..." RUN=<name> \
#     sbatch --account=<acct> slurm/stage_a_1node.sbatch
#SBATCH --job-name=fact-stage-a
#SBATCH --partition=boost_usr_prod
#SBATCH --account=CHANGE_ME
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1            # one torchrun; it spawns 4 workers
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
source "${SLURM_SUBMIT_DIR:?submit from the repository root}/slurm/stage_a.sbatch"
