import Pkg; Pkg.activate(@__DIR__)

using BioStructures, CUDA, Dates, Distributions, Flux, JLD2,
      LinearAlgebra, LogExpFunctions, Logging, LoggingExtras,
      NNlib, OneHotArrays, Random, Statistics
import Flux: Chain
using StatsBase: sample

include("src/data.jl")
include("src/model.jl")
include("src/train.jl")

println("=== includes ok ===")

Random.seed!(1)
N, K, K_atoms = 50, 8, 6
pos     = randn(Float32, 3, N)
n       = randn(Float32, 3, N); n ./= sqrt.(sum(n.^2; dims=1))
nuv     = nuv_from_n(n)
feats   = randn(Float32, 7, K_atoms, N)
nbh_ids = [mod1(i+j, N) for j in 1:K, i in 1:N]
window  = rand(Float32, K, N)
labels  = Float32.(rand(Bool, N))

model = DMasif(n_conv=3)
out   = model(pos, nuv, feats, nbh_ids, window)
println("forward pass ok — output size: ", size(out))

opt = Flux.setup(Adam(3f-4), model)
p   = ProteinData(pos, nuv, feats, nbh_ids, window, labels)
l   = train_step!(model, opt, p)
println("train_step! ok — loss: ", round(l; digits=4))

# ---- load_protein from a synthetic .jld2 ------------------------------------
mkpath("processed")
JLD2.jldsave("processed/FAKE.jld2";
    pos=pos, nuv=nuv, feats=feats, nbh_ids=nbh_ids, window=window, labels=labels)
pp = load_protein("FAKE"; processed_dir="processed")
println("load_protein ok — $(size(pp.pos, 2)) points, $(sum(pp.labels .== 1f0)) positives")
rm("processed/FAKE.jld2")

# ---- hash split sanity -------------------------------------------------------
ids  = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG", "HHH", "III", "JJJ"]
vals = filter(is_val, ids)
println("hash split ok — $(length(vals))/$(length(ids)) in val: $vals")

println("=== ALL TESTS PASSED ===")
