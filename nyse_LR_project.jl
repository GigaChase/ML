# Load Packages
using DataFrames
using CSV
using Plots
using Lathe
using GLM
using Statistics
using StatsPlots
using MLBase
using Missings



df = CSV.read("mystocks.csv", DataFrame)

describe(df)
scatter(df.open, df.close, xlabel = "Open", ylabel = "Close")

density(df.close, 
    title = "Density Plot",
    ylabel = "open",
    xlabel = "close",
    legend = false
)


using Lathe.preprocess: TrainTestSplit
train, test = TrainTestSplit(df, 0.75)


using GLM
fm = @formula(open ~ close)
linreg = lm(fm, train)

r2(linreg)




test_pred = predict(linreg, test) ;
train_pred = predict(linreg, train) ;


perf_test = df_original = DataFrame(y_original = test[!, :open], y_pred = test_pred)

perf_test.error = perf_test[!, :y_original] - perf_test[!, :y_pred]

perf_test.error_sq = perf_test.error.* perf_test.error ;

perf_train = df_original = DataFrame(y_original = train[!, :open], y_pred = train_pred)

perf_train.error = perf_train[!, :y_original] - perf_train[!, :y_pred]

perf_train.error_sq = perf_train.error.* perf_train.error ;


function mape(perf_df)
    mape = mean(abs.(perf_df.error./perf_df.y_original))
    return mape
end

function rmse(perf_df)
    rmse = sqrt(mean(perf_df.error.*perf_df.error))
    return rmse    
end



println("Mean Abs : ", mean(abs.(perf_test.error)), "\n")
println("Mean Abs % : ", mape(perf_test), "\n")

println("Root Mean2 : ", rmse(perf_test), "\n")
println("Mean Square : ", mean(perf_test.error_sq))


histogram(df.open, bins = 50, title = "Test Error Analysis", ylabel = "Frequency", xlabel = "Error", legend = false)





println("Mean Abs : ", mean(abs.(perf_train.error)), "\n")
println("Mean Abs % : ", mape(perf_train), "\n")

println("Root Mean2 : ", rmse(perf_train), "\n")
println("Mean Square : ", mean(perf_train.error_sq))


histogram(df.close, bins = 50, title = "Train Error Analysis", ylabel = "Frequency", xlabel = "Error", legend = false)




function cross_validation(train, k, fm = @formula(open ~ close))
    values = collect(Kfold(size(train)[1], k))

    for i in 1:k
        row = values[i]
        temp_train = train[row, :]
        temp_test = train[setdiff(1 : end, row), :]
        linreg = lm(fm, temp_train)
        perf_test = df_original = DataFrame(y_original = temp_test[!, :open], y_pred = predict(linreg, temp_test))
        perf_test.error = perf_test[!, :y_original] - perf_test[!, :y_pred]
            println("Mean Error for set $i : ", mean(abs.(perf_test.error)))
    end
end


cross_validation(train, 10)



# Multiple Linear Regression
fm1 = @formula(open ~ close + high + low)
linreg1 = lm(fm1, train)

r2(linreg1)

test_pred1 = predict(linreg1, test) ;
train_pred1 = predict(linreg1, train) ;


perf_test1 = df_original1 = DataFrame(y_original1 = test[!, :open], y_pred1 = test_pred1)

perf_test1.error = perf_test1[!, :y_original1] - perf_test1[!, :y_pred1]

perf_test1.error_sq = perf_test1.error.* perf_test1.error ;

perf_train1 = df_original1 = DataFrame(y_original1 = train[!, :open], y_pred1 = train_pred1)

perf_train1.error = perf_train1[!, :y_original1] - perf_train1[!, :y_pred1]

perf_train1.error_sq = perf_train1.error.* perf_train1.error ;



println("Mean Abs Test Error : ", mean(abs.(perf_train1.error)),"\n")

histogram(perf_test1.error, bins = 50, title = "Test Error Analysis", ylabel = "Frequency", xlabel = "Error", legend = false)

println("Mean Abs Test Error : ", mean(abs.(perf_test1.error)),"\n")

histogram(perf_train1.error, bins = 50, title = "Train Error Analysis", ylabel = "Frequency", xlabel = "Error", legend = false)
