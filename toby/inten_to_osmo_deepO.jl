
using HDF5, Lux, Metal, MLUtils, MultivariateStats, Optimisers, Random, Zygote, LinearAlgebra
using CairoMakie, Printf

# in : m×n, each column is an input function
# out: k×n, each column is an output function
# t  : k, each element is a time of the output
# every input is paired with a corresponding output column, over all the t values
struct DeepO_Data
    in::Matrix{Float32}
    out::Matrix{Float32}
    t::Vector{Float32}
    index::CartesianIndices{2}
end
function DeepO_Data(in, out, t)
    @assert size(in,2) == size(out,2)
    @assert length(t) == size(out,1)
    return DeepO_Data(in, out, t, CartesianIndices(out))
end

function Base.getindex(d::DeepO_Data, i::Int)
    idx = d.index[i]
    return ([d.in[:,idx[2]]; d.t[idx[1]]], d.out[idx])
end

function Base.getindex(d::DeepO_Data, i::AbstractVector{<:Integer})
    X = zeros(Float32, size(d.in,1)+1, length(i))
    y = zeros(Float32, 1, length(i))
    for (k,idx) in enumerate( d.index[i])
        X[:,k] .= [d.in[:,idx[2]]; d.t[idx[1]]]
        y[k] = d.out[idx]
    end
    return X, y
end

Base.length(d::DeepO_Data) = length(d.index)

function load_data(batchsize, train_split=0.8)
    datastore = h5open("trials6.h5", "r")
    data_I = read(datastore["I"])
    data_T = read(datastore["T"])
    data_c = read(datastore["c"])
    t = read(datastore["t"])
    close(datastore)

    (I_train, c_train), (I_test, c_test) = splitobs((data_I, data_c); at=train_split)

    return (
        DataLoader(DeepO_Data(I_train, c_train, t); batchsize, shuffle=true),
        DataLoader(DeepO_Data(I_test, c_test, t); batchsize, shuffle=false)
        )
end

function accuracy(loss, model, ps, st, dataloader)
    tse, total = 0, 0
    st = Lux.testmode(st)
    for (x, y) in dataloader
        predicted = first(model(x, ps, st))
        n = size(x,2)
        tse += n*loss(predicted, y)
        total += n
    end
    return tse / total
end

function train(loss, model, ps, st, train_loader, test_loader; num_epochs=10, kwargs...)
    train_state = Training.TrainState(model, ps, st, Adam(5.0f-3))

    ### Warm up the model
    x, y = first(train_loader)
    Training.compute_gradients(AutoZygote(), loss, (x, y), train_state)

    ### Train the model
    tr_acc, te_acc = 0.0, 0.0
    for epoch in 1:num_epochs
        stime = time()
        for (x, y) in train_loader
            gs, _, _, train_state = Training.single_train_step!(
                AutoZygote(), loss, (x, y), train_state)
        end
        ttime = time() - stime

        tr_acc = accuracy(loss,
            model, train_state.parameters, train_state.states, train_loader) * 100
        te_acc = accuracy(loss,
            model, train_state.parameters, train_state.states, test_loader) * 100

        @printf "[%2d/%2d] \t Time %.2fs \t Training Accuracy: %.2f%% \t Test Accuracy: \
                 %.2f%%\n" epoch num_epochs ttime tr_acc te_acc
    end

    return tr_acc, te_acc
end

function deepO(d_in, ntrunk)
    tb = Chain(
        Dense(1 + d_in => 2ntrunk, relu),
        Dense(2ntrunk => 2ntrunk, relu),
        Dense(2ntrunk => 2ntrunk),
    )
    x1 = x -> view(x, 1:ntrunk, :)
    x2 = x -> view(x, ntrunk+1:2ntrunk, :)
    return Chain( tb, WrappedFunction(x -> sum(x1(x).*x2(x), dims=1)) )
end

##
batch_size = 1000
train_loader, test_loader = load_data(batch_size)
model = deepO(101, 200)

##

device = cpu_device()
# train_loader = train_loader |> device
# test_loader = test_loader |> device
ps, st = Lux.setup(Xoshiro(0), model) |> device

loss = MSELoss()

##

train_acc, test_acc = train(loss, model, ps, st, device(train_loader), device(test_loader); num_epochs=4)

##
test_err = []
U = test_loader.data.in
V = test_loader.data.out
t = test_loader.data.t
st = Lux.testmode(st)
V_pred = similar(V)
for j in axes(U, 2)
    X = [repeat(U[:, j], outer=(1, length(t))); t']
    V_pred[:,j] .= vec(first(model(X, ps, st)))
end

test_err = map(norm, eachcol(V_pred - V))
fig,ax,_ = hist(log10.(test_err), bins=range(-2,1,31))
