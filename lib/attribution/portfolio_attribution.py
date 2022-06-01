# Define the weights and returns of all sectors in portfolio and benchmark
port_weights = [6.77, 8.52, 36.22, 5.24, 18.53, 14.46, 10.26]
bench_weights = [6.45, 8.99, 37.36, 4.65, 16.56, 18.87, 7.12]
port_returns = [-0.82, -3.28, 1.96, 0.44, 2.98, 2.32, 0.54]
bench_returns = [-0.73, -4.34, 1.98, 0.24, 2.22, -0.48, -0.42]

# Calculate and print the sum of weights of sector in portfolio
sum_port_weights = 0
for i in range(len(port_weights)):
    sum_port_weights += port_weights[i]

print('The sum of portfolio weights is: ', round(sum_port_weights))

# Calculate and printing the sum of weights of sector in benchmark
sum_bench_weights = 0
for i in range(len(bench_weights)):
    sum_bench_weights += bench_weights[i]

print('The sum of benchmark weights is: ', round(sum_bench_weights))

# Calculate and print the return from the portfolio
ret_port = 0
for i in range(len(port_returns)):
    ret_port += port_weights[i]*port_returns[i]/sum_port_weights

print('The return from the portfolio is: ', round(ret_port, 2))

# Calculate and print the return from the benchmark
ret_bench = 0
for i in range(len(bench_returns)):
    ret_bench += bench_weights[i]*bench_returns[i]/sum_bench_weights

print('The return from the benchmark is: ', round(ret_bench, 2))

# Calculat active return from the portfolio
ret_active = ret_port - ret_bench
print('The active return is: ', round(ret_active, 2))

# Calculating pure sector return
pure_sec_ret = 0
for i in range(len(port_weights)):
    pure_sec_ret += (port_weights[i]-bench_weights[i]) * \
        (bench_returns[i]-ret_bench)/sum_port_weights

print('The pure selection return is: ', round(pure_sec_ret, 2))

# Calculate selection interaction return
sel_int_ret = 0
for i in range(len(port_weights)):
    sel_int_ret += (port_weights[i]-bench_weights[i]) * \
        (port_returns[i]-bench_returns[i])/sum_port_weights

print('The selection interaction return is: ', round(sel_int_ret, 2))

# Calculate within-sector selection return
wit_sec_ret = 0
for i in range(len(port_weights)):
    wit_sec_ret += bench_weights[i] * \
        (port_returns[i]-bench_returns[i])/sum_port_weights
    
print('The within-sector selection return is: ', round(wit_sec_ret, 2))

# Sum up all the components
ret_active_2 = pure_sec_ret + sel_int_ret + wit_sec_ret
print('The total active return from the components is: ', round(ret_active_2, 2))