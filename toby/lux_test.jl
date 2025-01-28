using Lux, ADTypes, LuxCUDA, Optimisers, Printf, Random, Statistics, Zygote
using CairoMakie

function generate_data(rng::AbstractRNG)
    x = reshape(collect(range(-2.0f0, 2.0f0, 128)), (1, 128))
    y = evalpoly.(x, ((0, -2, 1),)) .+ randn(rng, Float32, (1, 128)) .* 0.1f0
    return (x, y)
end

rng = MersenneTwister()
Random.seed!(rng, 12345)

(x, y) = generate_data(rng)

model = Chain(Dense(1 => 16, relu), Dense(16 => 1))

opt = Adam(0.03f0)

const loss_function = MSELoss()

const dev_cpu = cpu_device()
const dev_gpu = gpu_device()

ps, st = Lux.setup(rng, model) |> dev_gpu

tstate = Training.TrainState(model, ps, st, opt)
vjp_rule = AutoZygote()

function main(tstate::Training.TrainState, vjp, data, epochs)
    data = data .|> gpu_device()
    for epoch in 1:epochs
        _, loss, _, tstate = Training.single_train_step!(vjp, loss_function, data, tstate)
        if epoch % 50 == 1 || epoch == epochs
            @printf "Epoch: %3d \t Loss: %.5g\n" epoch loss
        end
    end
    return tstate
end

tstate = main(tstate, vjp_rule, (x, y), 250)
y_pred = dev_cpu(Lux.apply(tstate.model, dev_gpu(x), tstate.parameters, tstate.states)[1])
