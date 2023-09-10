using Tensors

struct NeoHooke
    μ::Float64
    λ::Float64
end

struct MaterialState
    σ::Tensor{2, 3, Cdouble, 9} # std::array<Cdouble, 9>
end

function Ψ(C, mp::NeoHooke)
    μ = mp.μ
    λ = mp.λ
    Ic = tr(C)
    J = sqrt(det(C))
    return μ / 2 * (Ic - 3) - μ * log(J) + λ / 2 * log(J)^2
end

function constitutive_driver(F, mp::NeoHooke)
    C = tdot(F) # F' ⋅ F
    ∂²Ψ∂C², ∂Ψ∂C = Tensors.hessian(y -> Ψ(y, mp), C, :all)
    S = 2.0 * ∂Ψ∂C
    ∂S∂C = 2.0 * ∂²Ψ∂C²

    P = F ⋅ S
    I = one(S)
    ∂P∂F =  otimesu(I, S) + 2 * otimesu(F, I) ⊡ ∂S∂C ⊡ otimesu(F', I)

    return P, ∂P∂F
end;

function compute_mise(m::Ptr{Float64}, state::MaterialState)
    r = √(3/2 * dev(state.σ) ⊡ dev(state.σ))
    println("Mise in Julia: $(r)")
    unsafe_store!(m, r)
    return nothing
end


function do_assemble!(ge::Ptr{Float64}, ke::Ptr{Float64}, new_state::Ptr, prev_state::MaterialState, ∇u::Tensor{2, dim}, δuis::Ptr, ∇δuis::Ptr, ndofs, dΩ, mp::NeoHooke) where {dim}
    ge    = unsafe_wrap(Array, ge, ndofs)
    ke    = unsafe_wrap(Array, ke, (ndofs, ndofs))
    δuis  = unsafe_wrap(Array, δuis, ndofs)
    ∇δuis = unsafe_wrap(Array, ∇δuis, ndofs)

    do_assemble!(ge, ke, new_state, prev_state, ∇u, δuis, ∇δuis, ndofs, dΩ, mp)
end

function do_assemble!(ge::Vector{Float64}, ke::Matrix{Float64}, new_state::Ptr{MaterialState}, prev_state::MaterialState, ∇u::Tensor{2, dim}, δuis::Vector{<:Vec}, ∇δuis::Vector{<:Tensor{2}}, ndofs, dΩ::Float64, mp::NeoHooke) where {dim}
    # Compute deformation gradient F and right Cauchy-Green tensor C
    F = one(∇u) + ∇u
    C = tdot(F) # F' ⋅ F
    # Compute stress and tangent
    S, ∂S∂C = constitutive_driver(C, mp)
    P = F ⋅ S
    I = one(S)
    ∂P∂F = otimesu(I, S) + 2 * otimesu(F, I) ⊡ ∂S∂C ⊡ otimesu(F', I)

    # Loop over test functions
    for i in 1:ndofs
        # Test function and gradient
        δui = δuis[i]
        ∇δui = ∇δuis[i]
        # Add contribution to the residual from this test function
        ge[i] += ( ∇δui ⊡ P #=- δui ⋅ b =# ) * dΩ

        ∇δui∂P∂F = ∇δui ⊡ ∂P∂F # Hoisted computation
        for j in 1:ndofs
            ∇δuj = ∇δuis[j]
            # Add contribution to the tangent
            ke[i, j] += ( ∇δui∂P∂F ⊡ ∇δuj ) * dΩ
        end
    end

    # Store the Cauchy stress
    σ = P ⋅ F' / det(F)
    # vM = sqrt(3/2 * dev(σ) ⊡ dev(σ))
    unsafe_store!(new_state, MaterialState(σ))
    # println("Stress norm in julia: ", vM)
    return
end
