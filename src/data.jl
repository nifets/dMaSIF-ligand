# src/data.jl — atom data loading, SDF, surface sampling, neighbourhood helpers, preprocessing
# No `using` statements — assumed loaded by the entry point.

adpselector(res) = resnameselector(res, ["ADP"])

const ELEMENTS   = ["C", "H", "N", "O", "S", "SE"]
const ELEMENT_IDS = Dict([(ELEMENTS[i], i) for i in eachindex(ELEMENTS)])
const RADII       = [1.7, 1.1, 1.52, 1.55, 1.80, 1.90]  # van der Waals radii (Å)

# --------------------------------------------------------------------------- #
# Atom-level data extraction
# --------------------------------------------------------------------------- #

function get_atom_data(struc::StructuralElement)
    aminoacids = collectresidues(struc, standardselector)
    coords     = coordarray(aminoacids)
    types      = get.(Ref(ELEMENT_IDS), element.(collectatoms(aminoacids); strip=true), 2)

    adps       = collectresidues(struc, adpselector)
    adp_coords = coordarray(adps)

    # Sentinel: protein has no ADP residue
    if isempty(adp_coords)
        adp_coords = reshape(Float64[1e6, 1e6, 1e6], 3, 1)
    end
    return coords, types, adp_coords
end

# --------------------------------------------------------------------------- #
# Signed distance field (low-memory, O(A) per query point)
# --------------------------------------------------------------------------- #

function sdf(x, coords, radii; ideal_dist=1.05)
    a    = coords
    vecs = similar(a)
    d    = similar(a, 1, size(a, 2))
    v    = similar(d)
    res  = Float64[]
    for x_i in eachcol(x)
        vecs .= (x_i .- a).^2
        sum!(d, vecs)
        d .= .-sqrt.(d)
        v .= d ./ radii'
        L  = lsumexp(v)
        d .= exp.(d)
        σ  = dot(d, radii') / sum(d)
        push!(res, -σ * L)
    end
    return res .- ideal_dist
end

# ∇ₓ SDF
function grad_sdf(x, coords, radii)
    a     = coords
    vecs  = similar(a)
    d     = similar(a, 1, size(a, 2))
    ed    = similar(d)
    nd    = similar(d)
    grads = similar(x)

    for (i, x_i) in enumerate(eachcol(x))
        vecs .= (x_i .- a).^2
        sum!(d, vecs)
        d  .= .-sqrt.(d)
        nd .= d ./ radii'
        ed .= exp.(d)

        L = lsumexp(nd)
        ϕ = sum(ed)
        ψ = dot(ed, radii')
        σ = ψ / ϕ

        vecs .= exp.(nd) ./ (d .* radii') .* (x_i .- a)
        @views grads[:, i]  .= -σ * sum(vecs; dims=2) / exp(L)

        vecs .= (ed ./ d) .* (x_i .- a)
        @views grads[:, i] .-= -ψ * L * sum(vecs; dims=2) / ϕ^2

        vecs .*= radii'
        @views grads[:, i] .-= ϕ * L * sum(vecs; dims=2) / ϕ^2
    end
    return grads
end

# --------------------------------------------------------------------------- #
# Local reference frames (also used by model.jl)
# --------------------------------------------------------------------------- #

function nuv_from_n(n)
    @views x = n[1, :]; y = n[2, :]; z = n[3, :]
    s = sign.(z)
    a = -1f0 ./ (s .+ z)
    b = a .* x .* y

    u = similar(n)
    u[1, :] .= 1f0 .+ s .* a .* x .* x
    u[2, :] .= s .* b
    u[3, :] .= .-s .* x

    v = similar(n)
    v[1, :] .= b
    v[2, :] .= s .+ a .* y .* y
    v[3, :] .= .-y

    return cat(reshape(n, 3, 1, :), reshape(u, 3, 1, :), reshape(v, 3, 1, :); dims=2)
end

# --------------------------------------------------------------------------- #
# Surface sampling via gradient descent on squared SDF
# --------------------------------------------------------------------------- #

function sample_surface(coords, radii;
                        samples_per_atom=1, num_iters=10,
                        step_size=2.0, error_margin=0.3, batch_size=2000)
    A = size(coords, 2)
    B = samples_per_atom

    mysdf(p)      = sdf(p, coords, radii)
    mygrad_sdf(p) = grad_sdf(p, coords, radii)

    @debug "initial #samples $(A*B)"
    x  = rand(Normal(0.0, 1.0), 3, A, B)
    x .= x .* reshape(radii, 1, :, 1) .+ reshape(coords, 3, :, 1)
    x  = reshape(x, 3, :)

    batches = collect(Iterators.partition(axes(x, 2), batch_size))
    for i in 1:num_iters
        Threads.@threads for cols in batches
            @views x_b = x[:, cols]
            x_b .-= step_size .* mysdf(x_b)' .* mygrad_sdf(x_b)
        end
        @debug "SDF loss iter $i: $(mean(mysdf(x).^2) / 2)"
    end

    mask    = abs.(mysdf(x)) .< error_margin
    x       = x[:, mask]
    @debug "#samples after distance cull: $(count(mask))"

    normals = mygrad_sdf(x)
    foreach(normalize!, eachcol(normals))

    mask    = (mysdf(x .+ 4 .* normals) .- mysdf(x)) .> 0.5
    x       = x[:, mask]
    normals = normals[:, mask]
    @debug "#samples after trapped cull: $(count(mask))"

    grid_loc   = floor.(Int, x)
    unique_idx = unique(i -> grid_loc[:, i], 1:size(x, 2))
    x          = x[:, unique_idx]
    normals    = normals[:, unique_idx]
    @debug "#samples after subsampling: $(length(unique_idx))"

    return x, normals
end

# --------------------------------------------------------------------------- #
# k-nearest neighbours
# --------------------------------------------------------------------------- #

# Returns (ids::Matrix{Int}, dists::Matrix{Float}) for each column of x.
function knearest(x, coords; k=16)
    vecs    = similar(coords)
    sqdists = similar(coords, size(coords, 2))
    ids     = Matrix{Int}(undef, k, size(x, 2))
    dists   = similar(x, k, size(x, 2))
    for (i, x_i) in enumerate(eachcol(x))
        vecs    .= (x_i .- coords).^2
        sqdists .= dropdims(sum(vecs; dims=1); dims=1)
        @views ids[:, i]   .= sortperm(sqdists; alg=PartialQuickSort(k))[1:k]
        @views dists[:, i] .= sqrt.(sqdists[ids[:, i]])
    end
    return ids, dists
end

# --------------------------------------------------------------------------- #
# Geodesic distance helpers
# --------------------------------------------------------------------------- #

function dists_dots(x, n, nbh_ids)
    k     = size(nbh_ids, 1)
    vecs  = similar(x, 3, k)
    v     = similar(x, k)
    dists = similar(x, k, size(x, 2))
    dots  = similar(dists)
    for i in axes(x, 2)
        @views x_i = x[:, i]; n_i = n[:, i]
        vecs .= (x_i .- x[:, nbh_ids[:, i]]).^2
        v    .= dropdims(sum(vecs; dims=1); dims=1)
        @views dists[:, i] .= sqrt.(v)

        vecs .= n_i .* n[:, nbh_ids[:, i]]
        v    .= dropdims(sum(vecs; dims=1); dims=1)
        @views dots[:, i] .= v
    end
    return dists, dots
end

function quasi_geodesic_dist(x, n, nbh_ids; λ=1)
    dists, dots = dists_dots(x, n, nbh_ids)
    return dists .* (1 .+ λ .* (1 .- dots))
end

function gaussian_filter!(dists; σ=9)
    dists .= exp.(.-dists.^2 ./ (2 .* σ^2))
    return dists
end

# --------------------------------------------------------------------------- #
# Per-protein pipeline
# --------------------------------------------------------------------------- #

# Returns named tuple or nothing if the protein has no ADP.
function process_data_from_pdb(id::String;
                                atom_nbh_size=8, ligand_bind_range=8.0,
                                data_dir="data")
    struc                         = read(joinpath(data_dir, id * ".pdb"), PDBFormat)
    atom_coords, atom_types, adp_coords = get_atom_data(struc)

    # Skip early if BioStructures found no ADP residue (sentinel coords)
    if all(adp_coords .>= 1f5)
        @warn "$id: no ADP residue parsed — skipping"
        return nothing
    end

    atom_radii = RADII[atom_types]
    pos, normals = sample_surface(atom_coords, atom_radii; num_iters=10)

    nbh_atom_ids, atom_dists = knearest(pos, atom_coords; k=atom_nbh_size)
    sample_types  = reshape(atom_types[vec(nbh_atom_ids)], size(nbh_atom_ids))
    types_onehot  = onehotbatch(sample_types, 1:6, 2)
    inv_dists     = 1 ./ atom_dists

    # (7, atom_nbh_size, num_samples)
    feats  = cat(Float32.(types_onehot),
                 reshape(Float32.(inv_dists), 1, atom_nbh_size, :); dims=1)
    labels = knearest(pos, adp_coords; k=1)[2] .< ligand_bind_range

    return (pos=pos, normals=normals, feats=feats, labels=labels, adp_coords=adp_coords)
end

# Idempotent: skips already-cached proteins. Returns save path or nothing.
function preprocess_and_save(id::String;
                              processed_dir="processed",
                              data_dir="data",
                              kwargs...)
    outpath = joinpath(processed_dir, id * ".jld2")
    isfile(outpath) && (@debug "Skipping $id (cached)"; return outpath)

    result = process_data_from_pdb(id; data_dir=data_dir, kwargs...)
    isnothing(result) && return nothing

    n_pos = count(vec(result[:labels]))
    n_pts = size(result[:pos], 2)
    frac  = n_pos / n_pts
    min_adp_dist = round(minimum(knearest(result[:pos], result[:adp_coords]; k=1)[2]); digits=2)
    @info "$id: $n_pts surface pts, $n_pos pos, min_adp_dist=$(min_adp_dist)Å (frac=$(round(frac; digits=2)))"

    if n_pos < 10 || frac > 0.75
        @warn "Skipping $id: degenerate labels (n_pos=$n_pos, frac=$(round(frac; digits=2)), min_adp_dist=$(min_adp_dist)Å)"
        return nothing
    end

    pos     = Float32.(result[:pos])
    normals = Float32.(result[:normals])
    nuv     = nuv_from_n(normals)

    # Surface-surface neighbourhood for convolution
    nbh_ids, _ = knearest(pos, pos; k=16)
    window      = quasi_geodesic_dist(pos, normals, nbh_ids)
    gaussian_filter!(window)

    jldsave(outpath;
        pos     = pos,
        nuv     = nuv,
        feats   = Float32.(result[:feats]),
        nbh_ids = nbh_ids,
        window  = Float32.(window),
        labels  = vec(Float32.(result[:labels])))
    return outpath
end
