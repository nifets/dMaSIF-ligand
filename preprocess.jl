#!/usr/bin/env julia
# preprocess.jl — Fetch PDB IDs, download structures, preprocess to .jld2
# Usage: JULIA_NUM_THREADS=auto julia preprocess.jl

import Pkg; Pkg.activate(@__DIR__)

using BioStructures
using Dates
using Distributions
using HTTP
using JLD2
using JSON
using LinearAlgebra
using LogExpFunctions: logsumexp as lsumexp
using Logging
using LoggingExtras
using OneHotArrays
using Statistics

include("src/data.jl")

# --------------------------------------------------------------------------- #
# Logging setup
# --------------------------------------------------------------------------- #

const LOG_DIR = joinpath(@__DIR__, "logs")
mkpath(LOG_DIR)
mkpath(joinpath(@__DIR__, "data"))
mkpath(joinpath(@__DIR__, "processed"))
mkpath(joinpath(@__DIR__, "checkpoints"))

const LOG_PATH = joinpath(LOG_DIR, "preprocess_$(Dates.format(now(), "yyyymmdd_HHMMSS")).log")

global_logger(TeeLogger(
    ConsoleLogger(stderr, Logging.Info),
    FileLogger(LOG_PATH; always_flush=true)
))

@info "Logging to $LOG_PATH"
@info "Threads: $(Threads.nthreads()) / $(Sys.CPU_THREADS) logical cores"

# --------------------------------------------------------------------------- #
# PDB REST query
# --------------------------------------------------------------------------- #

"""
    fetch_adp_pdb_ids(; max_results=500, max_mw=50000.0) -> Vector{String}

Query RCSB for structures containing ADP (ligand ID ADP) that are X-ray
structures with resolution ≤ 2.5 Å and molecular weight ≤ `max_mw` Da.
"""
function fetch_adp_pdb_ids(; max_results=500, max_mw=50_000.0)
    query = Dict(
        "query" => Dict(
            "type"             => "group",
            "logical_operator" => "and",
            "nodes"            => [
                Dict(
                    "type"       => "terminal",
                    "service"    => "text",
                    "parameters" => Dict(
                        "attribute" => "rcsb_nonpolymer_instance_annotation.comp_id",
                        "operator"  => "exact_match",
                        "negation"  => false,
                        "value"     => "ADP"
                    )
                ),
                Dict(
                    "type"       => "terminal",
                    "service"    => "text",
                    "parameters" => Dict(
                        "attribute" => "exptl.method",
                        "operator"  => "exact_match",
                        "value"     => "X-RAY DIFFRACTION"
                    )
                ),
                Dict(
                    "type"       => "terminal",
                    "service"    => "text",
                    "parameters" => Dict(
                        "attribute" => "rcsb_entry_info.resolution_combined",
                        "operator"  => "less_or_equal",
                        "value"     => 2.5
                    )
                ),
                Dict(
                    "type"       => "terminal",
                    "service"    => "text",
                    "parameters" => Dict(
                        "attribute" => "rcsb_entry_info.polymer_entity_count",
                        "operator"  => "equals",
                        "value"     => 1
                    )
                ),
                Dict(
                    "type"       => "terminal",
                    "service"    => "text",
                    "parameters" => Dict(
                        "attribute" => "rcsb_entry_info.molecular_weight",
                        "operator"  => "less_or_equal",
                        "value"     => max_mw
                    )
                )
            ]
        ),
        "return_type"     => "entry",
        "request_options" => Dict(
            "paginate" => Dict("start" => 0, "rows" => max_results)
        )
    )

    url  = "https://search.rcsb.org/rcsbsearch/v2/query"
    body = JSON.json(query)
    resp = HTTP.post(url, ["Content-Type" => "application/json"], body)
    if resp.status != 200
        error("RCSB query failed with status $(resp.status)")
    end
    result = JSON.parse(String(resp.body))
    ids    = [entry["identifier"] for entry in result["result_set"]]
    @info "Fetched $(length(ids)) PDB IDs from RCSB"
    return ids
end

# --------------------------------------------------------------------------- #
# PDB download
# --------------------------------------------------------------------------- #

function download_pdb(id::String; outdir="data")
    path = joinpath(outdir, "$(id).pdb")
    isfile(path) && return path
    url  = "https://files.rcsb.org/download/$(id).pdb"
    resp = HTTP.get(url; status_exception=false)
    if resp.status != 200
        @warn "Could not download $id (HTTP $(resp.status))"
        return nothing
    end
    write(path, resp.body)
    return path
end

# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

function main()
    ids = fetch_adp_pdb_ids(; max_results=500, max_mw=50_000.0)

    n_ok    = 0
    n_skip  = 0
    n_fail  = 0

    for (i, id) in enumerate(ids)
        outpath = joinpath("processed", "$(id).jld2")
        if isfile(outpath)
            @debug "$id already preprocessed, skipping."
            n_skip += 1
            continue
        end

        pdb_path = download_pdb(id; outdir="data")
        isnothing(pdb_path) && (n_fail += 1; continue)

        t0 = time()
        try
            result = preprocess_and_save(id; data_dir="data", processed_dir="processed")
            elapsed = round(time() - t0; digits=1)
            if isnothing(result)
                @info "[$i/$(length(ids))] $id — skipped (no ADP or degenerate labels) ($(elapsed)s)"
                n_fail += 1
            else
                n_pts = JLD2.load(result, "pos") |> x -> size(x, 2)
                @info "[$i/$(length(ids))] $id — ok, $n_pts surface pts ($(elapsed)s) | total ok=$n_ok"
                n_ok += 1
            end
        catch e
            elapsed = round(time() - t0; digits=1)
            root = e
            while root isa CompositeException; root = root.exceptions[1]; end
            root isa TaskFailedException && (root = root.task.result)
            @warn "[$i/$(length(ids))] $id — FAILED ($(elapsed)s): $root"
            n_fail += 1
        end
    end

    @info "Done. ok=$n_ok  skipped=$n_skip  failed=$n_fail / total=$(length(ids))"
end

main()
