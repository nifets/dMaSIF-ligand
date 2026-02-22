#!/usr/bin/env julia
# run.jl — Overnight dMaSIF training with checkpointing and early stopping.
# Usage: JULIA_NUM_THREADS=auto julia run.jl

import Pkg; Pkg.activate(@__DIR__)

using BioStructures
using CUDA
using Dates
using Distributions
using Flux
using JLD2
using LinearAlgebra
using LogExpFunctions: logsumexp as lsumexp
using Logging
using LoggingExtras
using NNlib
using OneHotArrays
using Random
using Statistics
import Flux: Chain

include("src/data.jl")
include("src/model.jl")
include("src/train.jl")

# --------------------------------------------------------------------------- #
# Logging setup
# --------------------------------------------------------------------------- #

const LOG_DIR   = joinpath(@__DIR__, "logs")
const CKPT_PATH = joinpath(@__DIR__, "checkpoints", "checkpoint.jld2")
mkpath(LOG_DIR)
mkpath(joinpath(@__DIR__, "checkpoints"))

const LOG_PATH = joinpath(LOG_DIR, "training_$(Dates.format(now(), "yyyymmdd_HHMMSS")).log")

global_logger(TeeLogger(
    ConsoleLogger(stderr, Logging.Info),
    FileLogger(LOG_PATH; always_flush=true)
))

@info "Logging to $LOG_PATH"
@info "Threads: $(Threads.nthreads()) / $(Sys.CPU_THREADS) logical cores"

# --------------------------------------------------------------------------- #
# Sanity-check: at least one preprocessed protein exists before starting.
# --------------------------------------------------------------------------- #

if isempty(readdir("processed"))
    error("No preprocessed proteins found in processed/. Run preprocess.jl first.")
end

# --------------------------------------------------------------------------- #
# Checkpoint resume or fresh start
# --------------------------------------------------------------------------- #

if isfile(CKPT_PATH)
    @info "Resuming from checkpoint: $CKPT_PATH"
    ck          = JLD2.load(CKPT_PATH)
    start_epoch = ck["epoch"] + 1
    best_val    = ck["best_val"]

    model     = DMasif(n_conv=3)
    Flux.loadmodel!(model, ck["model_state"])
    opt_state = Flux.setup(Adam(3f-4), model)   # fresh optimiser (Adam momentum reset on resume)
    @info "Resumed at epoch $start_epoch, best_val=$(round(best_val; digits=4))"
else
    @info "Starting fresh training run."
    model     = DMasif(n_conv=3)
    opt_state = Flux.setup(Adam(3f-4), model)
    start_epoch = 1
    best_val    = Inf
end

# --------------------------------------------------------------------------- #
# Train — split is hash-based inside train!, rescans processed/ each epoch.
# --------------------------------------------------------------------------- #

try
    train!(model, opt_state;
           start_epoch     = start_epoch,
           best_val        = best_val,
           checkpoint_path = CKPT_PATH,
           processed_dir   = "processed",
           patience        = 30,
           save_every      = 5)
catch e
    @error "Training crashed" exception=(e, catch_backtrace())
    rethrow()
end
