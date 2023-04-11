#@title Optimized sdf functions
# low memory implementation of SDF
# to make faster - batch it up to enable parallelism
function sdf(x, coords, radii; ideal_dist = 1.05)
    a = coords
    vecs = similar(a)
    d = similar(a, 1, size(a, 2))
    v = similar(d)
    res = []
    for x_i in eachcol(x)
        vecs .= (x_i .- a).^2
        sum!(d, vecs)                               # d = sqdists(x_i, a)
        d .= .-sqrt.(d)                             # d = -dists(x_i, a)
        v .= d ./ radii'
        L = LogExpFunctions.logsumexp(v)            # L = logsumexp(-dists(x_i, a) / radii)
        d .= exp.(d)                                # d = exp.(-dists(x_i, a))
        σ = dot(d, radii') / sum(d)                 # σ = smoothed mean atom radius weighted by exp(-dists)
        push!(res, -σ*L)
    end
    return res .- ideal_dist
end

# memory heavy but correct
function heavy_sdf(x, coords, radii; ideal_dist = 1.05)
    sqdists = sum((reshape(x, 3, :, 1) .- reshape(coords, 3, 1, :)).^2; dims=1)
    dists = sqrt.(dropdims(sqdists; dims=1))
    expneg_dists= exp.(-dists)
    softavg_nbhrad = sum(expneg_dists .* radii'; dims=2) ./ sum(expneg_dists; dims=2)

    return vec(-softavg_nbhrad .* LogExpFunctions.logsumexp(-dists ./ radii'; dims=2) .- ideal_dist)
end

#∇ₓSDF - low memory - to make faster should operate in bigger batches of columns over x
function grad_sdf(x, coords, radii)
    a = coords
    vecs = similar(a)
    d = similar(a, 1, size(a, 2)) # -dists
    ed = similar(d) # exp(-dists)
    nd = similar(d) # -dists(x_i, a) / radii (normalized dists by radius)
    grads = similar(x)

    for (i, x_i) in enumerate(eachcol(x))
        vecs .= (x_i .- a).^2
        sum!(d, vecs)                               # d = sqdists(x_i, a)
        d .= .-sqrt.(d)                             # d = -dists(x_i, a)
        nd .= d ./ radii'                           # nd = -dists(x_i, a) / radii
        ed .= exp.(d)                               # ed = exp(-dists)

        L = LogExpFunctions.logsumexp(nd)           # L = logsumexp(-dists(x_i, a)/radii)
        ϕ = sum(ed)
        ψ = dot(ed, radii')
        σ = ψ / ϕ

        # add σ*∇L
        vecs .= exp.(nd)  ./ (d .* radii') .* (x_i .- a)
        @views grads[:, i] .= -σ * sum(vecs;dims=2) / exp(L)

        # add ∇ϕ component
        vecs .= (ed ./ d) .* (x_i .- a) # ∇ϕ component
        @views grads[:, i] .-= -ψ * L * sum(vecs;dims=2) / ϕ^2

        # add ∇ψ component
        vecs .*= radii' # ∇ψ component
        @views grads[:, i] .-= ϕ * L * sum(vecs;dims=2) / ϕ^2
    end

    return grads
end

# Profiling and correctness checks
function check_sdf()
    xx = rand(3, 4000)
    c = rand(3, 2000)
    r = rand(2000)

    println("My sdf")
    @time sdf(xx, c, r)
    println("Heavy sdf")
    @time heavy_sdf(xx, c, r)

    @assert isapprox(sdf(xx, c, r), heavy_sdf(xx, c, r))

    msd(x) = sum(heavy_sdf(x, c, r))
    println("My grad")
    @time grad_sdf(xx, c, r)
    println("Heavy grad")
    @time gradient(msd, xx)[1]

    @assert isapprox(grad_sdf(xx, c, r), gradient(msd, xx)[1])
end
