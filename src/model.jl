# src/model.jl — dMaSIF model layers and forward pass
# No `using` statements — assumed loaded by the entry point.
# nuv_from_n is defined in src/data.jl

# --------------------------------------------------------------------------- #
# Local reference frames
# --------------------------------------------------------------------------- #
# Fully functional (no mutation) so Zygote can differentiate w.r.t. frame.
function local_pos(pos, frame, nbh_ids)
    nbh_size, num_samples = size(nbh_ids)
    nbh_pos = reshape(pos[:, vec(nbh_ids)], 3, nbh_size, num_samples)
    ctr_pos = reshape(pos, 3, 1, num_samples)
    rel     = nbh_pos .- ctr_pos                                # (3, nbh_size, N)
    # frame^T @ rel via batched_mul: (3,3,N)^T x (3,nbh_size,N) -> (3,nbh_size,N)
    return NNlib.batched_mul(permutedims(frame, (2, 1, 3)), rel)
end

# --------------------------------------------------------------------------- #
# Chemical embedding layer
# --------------------------------------------------------------------------- #

struct ChemicalLayer
    atomMLP::Chain
    sampleMLP::Chain
end

function ChemicalLayer(; input_dim=7, hidden_dim=12, emb_dim=6)
    atomMLP   = Chain(Dense(input_dim => hidden_dim),
                      BatchNorm(hidden_dim, leakyrelu),
                      Dense(hidden_dim => emb_dim))
    sampleMLP = Chain(Dense(emb_dim => hidden_dim),
                      BatchNorm(hidden_dim, leakyrelu),
                      Dense(hidden_dim => emb_dim))
    return ChemicalLayer(atomMLP, sampleMLP)
end

function (m::ChemicalLayer)(x)
    input_dim, num_atoms, num_samples = size(x)
    x = reshape(x, input_dim, num_atoms * num_samples)
    x = m.atomMLP(x)                                # (emb_dim, num_atoms*N)
    x = reshape(x, :, num_atoms, num_samples)        # (emb_dim, num_atoms, N)
    x = dropdims(sum(x; dims=2); dims=2)             # (emb_dim, N)
    return m.sampleMLP(x)
end

Flux.@layer ChemicalLayer

# --------------------------------------------------------------------------- #
# Potential layer — scalar field over surface
# --------------------------------------------------------------------------- #

struct PotentialLayer
    chain::Chain
end

function PotentialLayer(; input_dim=6, hidden_dim=16)
    chain = Chain(Dense(input_dim => hidden_dim),
                  BatchNorm(hidden_dim, leakyrelu),
                  Dense(hidden_dim => hidden_dim),
                  BatchNorm(hidden_dim, leakyrelu),
                  Dense(hidden_dim => 1))
    return PotentialLayer(chain)
end

(m::PotentialLayer)(x) = m.chain(x)

Flux.@layer PotentialLayer

# --------------------------------------------------------------------------- #
# Geodesic convolution layer
# --------------------------------------------------------------------------- #

struct ConvLayer
    local_mlp::Chain
    emb_proj::Dense
end

function ConvLayer(; emb_dim=6)
    local_mlp = Chain(Dense(3 => 16),
                      BatchNorm(16, leakyrelu),
                      Dense(16 => 1))
    emb_proj  = Dense(emb_dim => emb_dim)
    return ConvLayer(local_mlp, emb_proj)
end

function (m::ConvLayer)(emb, window, p_ij, nbh_ids)
    feat_dim, nbh_size, num_samples = size(p_ij)
    emb_dim = size(emb, 1)

    p_flat  = reshape(p_ij, feat_dim, nbh_size * num_samples)
    w_ij    = reshape(m.local_mlp(p_flat), 1, nbh_size, num_samples)
    emb_ij  = reshape(emb[:, vec(nbh_ids)], emb_dim, nbh_size, num_samples)
    win     = reshape(window, 1, nbh_size, num_samples)

    out = dropdims(sum(win .* w_ij .* emb_ij; dims=2); dims=2)  # (emb_dim, N)
    return m.emb_proj(out)
end

Flux.@layer ConvLayer

# --------------------------------------------------------------------------- #
# Gauge update — aligns local frames via gradient of potential field
# --------------------------------------------------------------------------- #

function update_nuv(potentials, pos, nuv, nbh_ids, window)
    num_samples = size(pos, 2)
    nbh_size    = size(nbh_ids, 1)

    window   = reshape(window, 1, nbh_size, num_samples)
    ps       = potentials                                        # (1, N)
    pots_ij  = ps[:, nbh_ids] .- reshape(ps, 1, 1, num_samples) # (1, k, N)

    # Neighbour positions in local (u,v) tangent plane
    p_ij_uv = local_pos(pos, nuv[:, 2:3, :], nbh_ids)          # (2, k, N)

    # Weighted mean gradient direction in tangent plane
    new_u = dropdims(mean(window .* pots_ij .* p_ij_uv; dims=2); dims=2)  # (2, N)
    new_u = new_u ./ (sqrt.(sum(new_u.^2; dims=1) .+ 1f-8))               # normalise

    # Lift to 3D local coords, then rotate into global frame
    new_u = vcat(zeros(Float32, 1, num_samples), new_u)                    # (3, N)
    new_v = vcat(zeros(Float32, 1, num_samples), -new_u[3:3, :], new_u[2:2, :])

    new_uv = cat(reshape(new_u, 3, 1, :), reshape(new_v, 3, 1, :); dims=2) # (3, 2, N)
    new_uv = NNlib.batched_mul(nuv, new_uv)                                 # (3, 2, N)

    n = reshape(nuv[:, 1, :], 3, 1, :)
    return cat(n, new_uv; dims=2)                                           # (3, 3, N)
end

# --------------------------------------------------------------------------- #
# Full dMaSIF model
# --------------------------------------------------------------------------- #

struct DMasif
    chem_layer::ChemicalLayer
    potential_layer::PotentialLayer
    classifier_layer::Chain
    conv_layers::Vector{ConvLayer}
end

function DMasif(; n_conv=3)
    return DMasif(
        ChemicalLayer(),
        PotentialLayer(),
        Chain(Dense(6 => 16), BatchNorm(16, leakyrelu), Dense(16 => 1)),
        [ConvLayer(emb_dim=6) for _ in 1:n_conv]
    )
end

function (m::DMasif)(pos, nuv, feats, nbh_ids, window)
    emb        = m.chem_layer(feats)
    potentials = m.potential_layer(emb)
    nuv        = update_nuv(potentials, pos, nuv, nbh_ids, window)
    p_ij       = local_pos(pos, nuv, nbh_ids)
    for layer in m.conv_layers
        emb = layer(emb, window, p_ij, nbh_ids)
    end
    return m.classifier_layer(emb)
end

Flux.@layer DMasif
