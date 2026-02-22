# src/train.jl — training utilities: data loading, batching, loss, loop
# No `using` statements — assumed loaded by the entry point.

# --------------------------------------------------------------------------- #
# Protein data container
# --------------------------------------------------------------------------- #

struct ProteinData
    pos::Matrix{Float32}        # (3, N)
    nuv::Array{Float32, 3}      # (3, 3, N)
    feats::Array{Float32, 3}    # (feat_dim, K_atoms, N)
    nbh_ids::Matrix{Int}        # (nbh_size, N)
    window::Matrix{Float32}     # (nbh_size, N)
    labels::Vector{Float32}     # (N,)
end

function load_protein(id::String; processed_dir="processed")
    path = joinpath(processed_dir, "$(id).jld2")
    d = JLD2.load(path)
    return ProteinData(
        d["pos"],
        d["nuv"],
        d["feats"],
        d["nbh_ids"],
        d["window"],
        d["labels"]
    )
end

# Deterministic 10% val split: reproducible from the ID alone, no state needed.
is_val(id::String) = hash(id) % UInt(10) == 0

# Rescan processed_dir and split into trn/val via hash.
function scan_ids(processed_dir="processed")
    all_ids = [splitext(basename(f))[1]
               for f in readdir(processed_dir; join=false)
               if endswith(f, ".jld2")]
    val_ids = filter(is_val, all_ids)
    trn_ids = filter(!is_val, all_ids)
    return trn_ids, val_ids
end

# --------------------------------------------------------------------------- #
# Augmentation
# --------------------------------------------------------------------------- #

function random_rotation()
    q = randn(Float32, 4)
    q ./= norm(q)
    a, b, c, d = q
    return Float32[
        a^2+b^2-c^2-d^2  2(b*c-a*d)        2(b*d+a*c);
        2(b*c+a*d)        a^2-b^2+c^2-d^2   2(c*d-a*b);
        2(b*d-a*c)        2(c*d+a*b)         a^2-b^2-c^2+d^2
    ]
end

# --------------------------------------------------------------------------- #
# Loss
# --------------------------------------------------------------------------- #

function balanced_idx(labels::AbstractVector)
    pos = findall(==(1.0f0), labels)
    neg = findall(==(0.0f0), labels)
    n   = min(length(pos), length(neg))
    return vcat(sample(pos, n; replace=false), sample(neg, n; replace=false))
end

function balanced_bce(affinity::AbstractMatrix, labels::AbstractVector, idx::AbstractVector)
    # affinity: (1, N), labels: (N,)
    logits = vec(affinity)[idx]
    y      = labels[idx]
    return mean(Flux.logitbinarycrossentropy.(logits, y))
end

# --------------------------------------------------------------------------- #
# Per-sample training step
# --------------------------------------------------------------------------- #

function train_step!(model, opt_state, p::ProteinData)
    R   = random_rotation()
    pos = R * p.pos
    # Rotate all nuv columns at once: R*(3,3N) → reshape back to (3,3,N)
    nuv = reshape(R * reshape(p.nuv, 3, :), 3, 3, size(p.nuv, 3))

    idx  = balanced_idx(p.labels)
    loss, grads = Flux.withgradient(model) do m
        balanced_bce(m(pos, nuv, p.feats, p.nbh_ids, p.window), p.labels, idx)
    end
    Flux.update!(opt_state, model, grads[1])
    return loss
end

# --------------------------------------------------------------------------- #
# Validation — no gradient, no augmentation
# --------------------------------------------------------------------------- #

function val_loss(model, id::String; processed_dir="processed")
    return val_loss(model, load_protein(id; processed_dir))
end

function val_loss(model, p::ProteinData)
    idx = balanced_idx(p.labels)
    aff = model(p.pos, p.nuv, p.feats, p.nbh_ids, p.window)
    return balanced_bce(aff, p.labels, idx)
end

# --------------------------------------------------------------------------- #
# Training loop with early stopping and checkpointing
# --------------------------------------------------------------------------- #

"""
    train!(model, opt_state; kwargs...)

Train indefinitely, rescanning `processed_dir` each epoch for new proteins.
Split is hash-based (deterministic 90/10), so new proteins land in the correct
set without ever contaminating validation.

- When val proteins are available: checkpoint on improvement, early-stop after `patience`.
- When val set is empty: checkpoint every `save_every` epochs as a fallback.

Keyword arguments:
- `start_epoch=1`
- `best_val=Inf`
- `checkpoint_path="checkpoints/checkpoint.jld2"`
- `processed_dir="processed"`
- `patience=30`
- `save_every=5`
"""
function train!(model, opt_state;
                start_epoch=1,
                best_val=Inf,
                checkpoint_path="checkpoints/checkpoint.jld2",
                processed_dir="processed",
                patience=30,
                save_every=5)

    epochs_no_improve = 0
    epoch = start_epoch
    mkpath(dirname(checkpoint_path))

    while true
        # Rescan processed dir — picks up proteins added by preprocess.jl mid-run.
        trn_ids, val_ids = scan_ids(processed_dir)
        @debug "Epoch $epoch: $(length(trn_ids)) train, $(length(val_ids)) val proteins"

        losses = Float32[]
        for id in shuffle(trn_ids)
            p = try
                load_protein(id; processed_dir)
            catch e
                @warn "Skipping $id — failed to load: $e"
                continue
            end
            if sum(p.labels .== 1f0) == 0 || sum(p.labels .== 0f0) == 0
                @debug "Skipping $id — no balanced labels"
                continue
            end
            l = train_step!(model, opt_state, p)
            push!(losses, l)
        end

        mean_trn = isempty(losses) ? NaN32 : mean(losses)

        # Validation
        Flux.testmode!(model)
        val_losses = Float32[]
        for id in val_ids
            p = try
                load_protein(id; processed_dir)
            catch e
                @warn "Val skip $id — $e"
                continue
            end
            if sum(p.labels .== 1f0) == 0 || sum(p.labels .== 0f0) == 0
                continue
            end
            push!(val_losses, val_loss(model, p))
        end
        Flux.trainmode!(model)
        mean_val = isempty(val_losses) ? NaN32 : mean(val_losses)

        @info "Epoch $epoch | trn=$(round(mean_trn; digits=4)) | val=$(round(mean_val; digits=4)) | no_improve=$epochs_no_improve | n_trn=$(length(trn_ids)) n_val=$(length(val_ids))"

        function save_checkpoint(; tag="")
            JLD2.jldsave(checkpoint_path;
                model_state = Flux.state(model),
                epoch       = epoch,
                best_val    = best_val)
            @info "Checkpoint saved$(tag)"
        end

        if !isnan(mean_val)
            # Normal path: checkpoint on val improvement, early-stop on patience.
            if mean_val < best_val
                best_val = mean_val
                epochs_no_improve = 0
                save_checkpoint(; tag=" (val=$(round(best_val; digits=4)))")
            else
                epochs_no_improve += 1
                if epochs_no_improve >= patience
                    @info "Early stopping after $patience epochs without improvement."
                    break
                end
            end
        else
            # No val proteins yet — save periodically.
            if epoch % save_every == 0
                save_checkpoint(; tag=" (periodic, no val set)")
            end
        end

        epoch += 1
    end
end
