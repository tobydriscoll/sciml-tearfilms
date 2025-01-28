
using HDF5, Lux, Metal, MLUtils, MultivariateStats, Optimisers, Random, Zygote, LossFunctions,LinearAlgebra
using CairoMakie, Printf, Statistics

function load_data()
    # datastore = h5open("trials6_300.h5", "r")
    data_I = []
    data_T = []
    data_c = []
    for file in ["/Users/driscoll/Dropbox/research/tearfilm/thermal/inverse/trials_many_alldata_v3.h5"]
        datastore = h5open(file, "r")
        if isempty(data_I)
            data_I = read(datastore["I"])
            data_T = read(datastore["T"])
            data_c = read(datastore["c"])
        else
            data_I = [data_I read(datastore["I"])]
            data_T = [data_T read(datastore["T"])]
            data_c = [data_c read(datastore["c"])]
        end
        # t = read(datastore["t"])
        close(datastore)
    end
    return (; I=data_I, T=data_T, c=data_c)
end

function convert_data(data, batchsize, train_split=0.8, n_comp=20)
    pcac = fit(PCA, data.c, maxoutdim=n_comp, pratio=0.9999);

    X = [data.I; data.T];
    y = predict(pcac, data.c);
    (X_train, y_train), (X_test, y_test) = splitobs((X, y); at=train_split)
    t = range(0, 1, size(data.I, 1))
    return (
        t, pcac,
        DataLoader(collect.((X_train, y_train)); batchsize, shuffle=false),
        DataLoader(collect.((X_test, y_test)); batchsize, shuffle=false))
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

function train(loss, model, ps, st, train_loader, test_loader; num_epochs=10, rate=0.005, kwargs...)
    train_state = Training.TrainState(model, ps, st, Adam(Float32(rate)))

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
            model, train_state.parameters, train_state.states, train_loader)
        te_acc = accuracy(loss,
            model, train_state.parameters, train_state.states, test_loader)

        @printf "[%2d/%2d] \t Time %.2fs \t Training loss: %.4g \t Test loss: \
                 %.4g\n" epoch num_epochs ttime tr_acc te_acc
    end

    return tr_acc, te_acc
end

##
batch_size = 1000

data = load_data()
t, pcac, train_loader, test_loader = convert_data(data, batch_size)
d_input = size(train_loader.data[1], 1)
d_output = size(train_loader.data[2], 1)
model = Chain(
    Dense(d_input => 512, relu),
    Dense(512 => 256, relu),
    Dense(256 => 128, relu),
    Dense(128 => d_output),
)

##

device = cpu_device()
# train_loader = train_loader |> device
# test_loader = test_loader |> device
ps, st = Lux.setup(Xoshiro(0), model) |> device
loss = MSELoss()

##

train_acc, test_acc = train(loss, model, ps, st, device(train_loader), device(test_loader); num_epochs=300, rate=0.0004)

##

test_err = []
for (X, y) in test_loader
    predicted = first(model(X, ps, st))
    c_pred = reconstruct(pcac, predicted )
    c_true = reconstruct(pcac, y)
    append!(test_err, map(norm, eachcol(c_pred-c_true)) / sqrt(length(t) - 1))
    # append!(test_err, [abs(maximum(y)-maximum(z)) for (y,z) in  zip(eachcol(c_pred),eachcol(c_true))])
end

@show quantile(log10.(test_err),0:.25:1)

fig,ax,_ = hist(log10.(test_err), bins=range(-2,0.5,31))

##
(X,y) = first(test_loader)
predicted = first(model(X, ps, st))
a = predicted[:,379]; b =y[:,379]
# a[10:end].=0
# a[6:end] .= b[6:end]
fig = lines(t,reconstruct(pcac,a))
lines!(t, reconstruct(pcac,b))
fig

##
test_err = []
for (X, y) in test_loader
    predicted = first(model(X, ps, st))
    c_pred = reconstruct(pcac, predicted )
    c_true = reconstruct(pcac, y)
    append!(test_err, [abs(maximum(y)-maximum(z)) for (y,z) in  zip(eachcol(c_pred),eachcol(c_true))])
end
fig,ax,_ = hist(log10.(test_err), bins=range(-3,0,31))
