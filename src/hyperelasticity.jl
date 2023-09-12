using Tensors

# Material parameters
struct NeoHooke
    μ::Float64 # double
    λ::Float64 # double
end

# Material state (not needed for this model but included as an example)
struct MaterialState
    σ::Tensor{2, 3, Float64, 9} # std::array<double, 9>
end

# Potential energy density
function Ψ(C, mp::NeoHooke)
    μ = mp.μ
    λ = mp.λ
    Ic = tr(C)
    J = sqrt(det(C))
    return μ / 2 * (Ic - 3) - μ * log(J) + λ / 2 * log(J)^2
end

# Material routine: Strain in, stress and tangent out
function constitutive_driver(C, mp::NeoHooke)
    # Automatic differentiation to compute second and first derivative
    ∂²Ψ∂C², ∂Ψ∂C = Tensors.hessian(y -> Ψ(y, mp), C, :all)
    S = 2.0 * ∂Ψ∂C # Second Piola-Kirchoff stress
    ∂S∂C = 2.0 * ∂²Ψ∂C²
    return S, ∂S∂C
end

# Compute effective von Mise stress
function compute_mise(state::MaterialState)
    r = √(3/2 * dev(state.σ) ⊡ dev(state.σ))
    return r
end

# Assembly routine compute the contribution to the local residual vector and
# the local tangent matrix for one quadrature point
function do_assemble!(
        ge::Vector{Float64}, ke::Matrix{Float64},
        new_state::Ptr{MaterialState}, prev_state::MaterialState,
        ∇u::Tensor{2}, δu::Vector{<:Vec}, ∇δu::Vector{<:Tensor{2}},
        ndofs, dΩ::Float64, mp::NeoHooke
    )

    # Compute deformation gradient F and right Cauchy-Green tensor C
    F = one(∇u) + ∇u
    C = tdot(F) # F' ⋅ F

    # Compute stress and tangent
    S, ∂S∂C = constitutive_driver(C, mp)

    # Convert to first Piola-Kirchoff stress
    P = F ⋅ S
    I = one(S)
    ∂P∂F = otimesu(I, S) + 2 * otimesu(F, I) ⊡ ∂S∂C ⊡ otimesu(F', I)

    # Loop over test functions
    for i in 1:ndofs
        # Add contribution to the residual vector
        ge[i] += ( ∇δu[i] ⊡ P #=- δu[i] ⋅ b =# ) * dΩ
        for j in 1:ndofs
            # Add contribution to the tangent matrix
            ke[i, j] += ( ∇δu[i] ⊡ ∂P∂F ⊡ ∇δu[j] ) * dΩ
        end
    end

    # Store the Cauchy stress as material state
    σ = P ⋅ F' / det(F)
    unsafe_store!(new_state, MaterialState(σ))

    return nothing
end

# Entry point from C
function do_assemble!(
    #             Type in Julia                     Type in C
    ge         :: Ptr{Float64},                   # double*
    ke         :: Ptr{Float64},                   # double*
    new_state  :: Ptr{MaterialState},             # MaterialState*
    prev_state :: MaterialState,                  # MaterialState
    ∇u         :: Tensor{2, dim, Float64},        # std::array<double, dim * dim>
    δuis       :: Ptr{Vec{dim, Float64}},         # std::array<double, dim>*
    ∇δuis      :: Ptr{<:Tensor{2, dim, Float64}}, # std::array<double, dim * dim>*
    ndofs      :: Int32,                          # int
    dΩ         :: Float64,                        # double
    mp         :: NeoHooke,                       # NeoHooke
) where {dim}

    # Note: Using the package UnsafeArrays.jl avoids the alloc and GC of the
    #       Array metadata (the data itself will be shared regardless).
    ge    = unsafe_wrap(Array, ge, ndofs)
    ke    = unsafe_wrap(Array, ke, (ndofs, ndofs))
    δuis  = unsafe_wrap(Array, δuis, ndofs)
    ∇δuis = unsafe_wrap(Array, ∇δuis, ndofs)
    do_assemble!(ge, ke, new_state, prev_state, ∇u, δuis, ∇δuis, ndofs, dΩ, mp)
end
