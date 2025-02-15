import math
import copy

activation_function = {
    "ReLU": "ReLU",
    "Linear": "Linear",
    "Sigmoid": "Sigmoid",
    "Softmax": "Softmax",
}

loss_function = {
    "mean_squared_error": "mean_squared_error",
    "binary_cross_entropy": "binary_cross_entropy",
    "categorical_cross_entropy": "categorical_cross_entropy",
}

class Network:
    def __init__(
        self,
        weights,
        biasesWeights,
        loss_function,
        output_activation_function=activation_function["Linear"],
        other_activation_function  = {'all':activation_function["ReLU"]},
    ):
        self.weights = copy.deepcopy(weights)
        self.new_weights_gradient = copy.deepcopy(weights)
        self.neuron_gradient = []
        self.biasesWeights = copy.deepcopy(biasesWeights)
        self.new_biasesWeights_gradient = copy.deepcopy(biasesWeights)
        self.biases = 1
        self.output_activation_function = output_activation_function
        self.other_activation_function = other_activation_function
        self.loss_function = loss_function
        self.output_array = []


    def _Linear(self, x):
        return x

    def _ReLU(self, x):
        if isinstance(x, (list, tuple)):
            return [max(0, xi) for xi in x]
        else:
            return max(0, x)

    def _Sigmoid(self, x):
        # 修改過的
        if isinstance(x, (list, tuple)):
            return [1 / (1 + math.exp(-xi)) for xi in x]
        else:
            return 1 / (1 + math.exp(-x))

    def _Softmax(self, x):
        max_x = max(x)
        exp_values = [math.exp(xi - max_x) for xi in x]
        sum_exp_values = sum(exp_values)
        softmax_values = [exp_val / sum_exp_values for exp_val in exp_values]
        return softmax_values

    def mean_squared_error(self, outputs, expected_outputs):
        mse = sum(
            (expect - output) ** 2 for output, expect in zip(outputs, expected_outputs)
        ) / len(outputs)
        self.loss_function = 'mean_squared_error'
        return mse

    def binary_cross_entropy(self, outputs, expected_outputs):
        bce = -sum(
            expected_output * math.log(output)
            + (1 - expected_output) * math.log(1 - output)
            for output, expected_output in zip(outputs, expected_outputs)
        )
        self.loss_function = 'binary_cross_entropy'
        return bce

    def categorical_cross_entropy(self, outputs, expected_outputs):
        ce = -sum(
            expected_output * math.log(output)
            for output, expected_output in zip(outputs, expected_outputs)
        )
        self.loss_function = 'categorical_cross_entropy'
        return ce

    def _check_activation_function(self, prev_input,active_function):
        if active_function == "Linear":
            prev_input = self._Linear(prev_input)
        elif active_function == "ReLU":
            prev_input = self._ReLU(prev_input)
        elif active_function == "Sigmoid":
            prev_input = self._Sigmoid(prev_input)
        elif active_function == "Softmax":
            prev_input = self._Softmax(prev_input)
        return prev_input
    def forward(self, inputs):
        prev_input = inputs
        layer_len = len(self.weights)
        self.output_array = []
        self.output_array.append(inputs)
        for layer in range(layer_len):
            row_len = len(self.weights[layer])
            init_output = [0] * row_len
            for array in range(row_len):
                for column in range(len(self.weights[layer][array])):
                    weight = self.weights[layer][array][column]
                    if weight and prev_input[column]:
                        init_output[array] += weight * prev_input[column]

                for column in range(len(self.biasesWeights[layer][array])):
                    init_output[array] += self.biasesWeights[layer][array][column] * self.biases

                # Check for other activation functions first
                if self.other_activation_function.get(layer, False):
                    init_output[array] = self._check_activation_function(init_output[array], self.other_activation_function[layer])
                # Apply ReLU for all other layers except the output layer
                elif layer != layer_len - 1:
                    init_output[array] = self._ReLU(init_output[array])
                # Apply the output activation function for the output layer
                elif layer == layer_len - 1:
                    init_output[array] = self._check_activation_function(init_output[array], self.output_activation_function)

            self.output_array.append(init_output)
            # Store the output of the current layer as the input for the next layer
            prev_input = init_output

        return prev_input
                    
    def backward(self, expected_outputs):
        # 獲取神經網路層數
        layer_len = len(self.output_array)
        # 初始化空白的神經網絡節點
        self.neuron_gradient = [
            [{'total_output_gradient': 0, 
                'output_netOutput_gradient': 0,} 
                for _ in range(len(self.output_array[layer]))]
            for layer in range(layer_len)
        ]

        # 紀錄每一層的derivative -> 對 total-output output-netOutput netOutput-weight
        for layer in range(layer_len - 1, -1, -1):
            for column in range(len(self.output_array[layer])):
              total_output_gradient = 0
              output_netOutput_gradient = 0
              # netOutput_weight_gradient = 0
              # 累計相乘(連鎖反應)每一層output對expect的影響
              # total_output_all = 0
              
              if layer == layer_len - 1:
                # 對輸出層-損失函數的處理
                if self.loss_function == 'mean_squared_error':
                    total_output_gradient = (2 / len(self.output_array[layer])) * (self.output_array[layer][column] - expected_outputs[column])
                elif self.loss_function == 'binary_cross_entropy':
                    expect = expected_outputs[column]
                    output = self.output_array[layer][column]
                    total_output_gradient = -((expect / output) - (1 - expect) / (1 - output))
                
                self.neuron_gradient[layer][column]['total_output_gradient'] = total_output_gradient

                # 對激活函數的處理
                if self.output_activation_function == activation_function['Linear']:
                    output_netOutput_gradient = 1
                elif self.output_activation_function == activation_function['Sigmoid']:
                    output = self.output_array[layer][column]
                    output_netOutput_gradient = output * (1 - output)

                self.neuron_gradient[layer][column]['output_netOutput_gradient'] = output_netOutput_gradient

                # 累計相乘(連鎖反應)每一層output對expect的影響，開始累計到上一層中
                total_output_all = total_output_gradient * output_netOutput_gradient
                # 避免重複跑
                if column == 0:
                    for array in range(len(self.neuron_gradient[layer-1])):
                        self.neuron_gradient[layer-1][array]['total_output_gradient'] = total_output_all

                # 對權重的處理，上一層的output值
                for array in range(len( self.output_array[layer - 1])):
                    self.new_weights_gradient[layer-1][column][array] = self.output_array[layer - 1][array]

                # 對bias權中的處理，上一層的output值
                self.new_biasesWeights_gradient[layer-1][column][0] = self.biases


              # 對隱藏層的處理,影響到下一層的幾個神經元
              elif layer > 0:
                # 影響到下一層的幾個神經元
                for array in range(len(self.output_array[layer+1])):
              
                  # 對激活函數的處理
                  if self.other_activation_function.get(layer, None):
                    if self.other_activation_function[layer] == activation_function['Linear']:
                        output_netOutput_gradient = 1
                    elif self.other_activation_function[layer] == activation_function['Sigmoid']:
                        output = self.output_array[layer][column]
                        output_netOutput_gradient = output * (1 - output)
                  # 其餘預設ReLu
                  else:
                    if self.output_array[layer][column] > 0:
                        output_netOutput_gradient = 1
                    else:
                        output_netOutput_gradient = 0

                  self.neuron_gradient[layer][column]['output_netOutput_gradient'] = output_netOutput_gradient

                # 進行累積相乘和相加所有關聯到下一層神經元
                # 因為output_array放入剛開始手動輸入的input 所以比weight 多了一層
                total_output_all = 0
                # 有bug if layer - 1 > 0:
                if layer -1 >= 0:
                  for array in range(len(self.weights[layer])):
                    next_net_now_output = self.weights[layer][array][column]
                    total_output_gradient = self.neuron_gradient[layer+1][array]['total_output_gradient']
                    output_netOutput_gradient = self.neuron_gradient[layer+1][array]['output_netOutput_gradient']
                    total_output_one = total_output_gradient * output_netOutput_gradient * next_net_now_output
                    # 所有關聯神經元相加
                    total_output_all += total_output_one
                    self.neuron_gradient[layer][column]['total_output_gradient'] = total_output_all

                  # 要將相乘紀錄到上一層神經元中，並且多乘入output_netOutput_gradient
                  self.neuron_gradient[layer-1][column]['total_output_gradient'] = total_output_all * output_netOutput_gradient

                # 對權重的處理，上一層的output值
                # for array in range(len( self.output_array[layer - 1])):
                for array in range(len( self.new_weights_gradient[layer - 1])):
                    for id in range(len( self.new_weights_gradient[layer - 1][array])):
                        self.new_weights_gradient[layer-1][array][id] = self.output_array[layer - 1][id]

                # 對bias權中的處理，上一層的output值
                self.new_biasesWeights_gradient[layer-1][column][0] = self.biases


    def zero_grad(self, learning_rate):
        for layer in range(len(self.new_weights_gradient)):
            for array in range(len(self.new_weights_gradient[layer])):
                # 避開第一次手動輸入的input值
                total_output_gradient = self.neuron_gradient[layer+1][array]['total_output_gradient']
                output_netOutput_gradient =  self.neuron_gradient[layer+1][array]['output_netOutput_gradient']
                for column in range(len(self.new_weights_gradient[layer][array])):
                    old_weight = self.weights[layer][array][column]
                    weight_gradient = self.new_weights_gradient[layer][array][column]
                    gradient = total_output_gradient * output_netOutput_gradient * weight_gradient         
                    new_weight = old_weight - learning_rate * gradient
                    self.weights[layer][array][column] = new_weight

                bias_gradient = self.new_biasesWeights_gradient[layer][array][0]
                gradient =  total_output_gradient * output_netOutput_gradient * bias_gradient
                old_bias_weight = self.biasesWeights[layer][array][0]
                new_bias_weight = old_bias_weight - learning_rate * gradient
                self.biasesWeights[layer][array][0] = new_bias_weight
                    

print("--------------Model01------------------")
network_1_1 = Network(
    [
        [
            [0.5, 0.2], 
            [0.6, -0.6]
        ], 
        [
            [0.8, -0.5], 
        ],
        [
            [0.6],
            [-0.3],
        ]
    ],
    [
        [
            [0.3], 
            [0.25]
        ], 
        [
            [0.6], 
        ],
        [
            [0.4], 
            [0.75]
        ]
    ],
    loss_function = 'mean_squared_error',
    other_activation_function = {1:activation_function["Linear"]}
)

network_outputs = network_1_1.forward([1.5, 0.5])
network_expected_outputs = [0.8, 1]
print(
    "Total Loss",
    network_1_1.mean_squared_error(
        network_outputs, network_expected_outputs
    ),
)


print("--------------task1-1------------------")
network_1_1.backward(network_expected_outputs)
network_1_1.zero_grad(0.01)
for layer in range(len(network_1_1.weights)):
        print('layer',layer)
        print(network_1_1.weights[layer])
        print(network_1_1.biasesWeights[layer])

print("--------------task1-2------------------")
network_1_2 = Network(
    [
        [
            [0.5, 0.2], 
            [0.6, -0.6]
        ], 
        [
            [0.8, -0.5], 
        ],
        [
            [0.6],
            [-0.3],
        ]
    ],
    [
        [
            [0.3], 
            [0.25]
        ], 
        [
            [0.6], 
        ],
        [
            [0.4], 
            [0.75]
        ]
    ],
    loss_function = 'mean_squared_error',
    other_activation_function = {1:activation_function["Linear"]}
)
network_outputs = 0
network_expected_outputs = [0.8, 1]
for i in range(1000):
    network_outputs = network_1_2.forward([1.5, 0.5])
    network_1_2.backward(network_expected_outputs)
    network_1_2.zero_grad(0.01)

print('Total Loss',network_1_2.mean_squared_error(network_outputs, network_expected_outputs))


print("--------------Model02------------------")
network_2_1 = Network(
    [
        [
            [0.5, 0.2], 
            [0.6, -0.6]
        ], 
        [
            [0.8, 0.4], 
        ],
    ],
    [
        [
            [0.3], 
            [0.25]
        ], 
        [
            [-0.5]
        ]
    ],
    loss_function = 'binary_cross_entropy',
    output_activation_function=activation_function["Sigmoid"],
)

network_outputs = network_2_1.forward([0.75, 1.25])
network_expected_outputs = [1]
print(
    "Total Loss",
    network_2_1.binary_cross_entropy(
        network_outputs, network_expected_outputs
    ),
)

print("--------------task2-1------------------")
network_2_1.backward(network_expected_outputs)
network_2_1.zero_grad(0.1)
for layer in range(len(network_2_1.weights)):
        print('layer',layer)
        print(network_2_1.weights[layer])
        print(network_2_1.biasesWeights[layer])

print("--------------task2-2------------------")
network_2_2 = Network(
    [
        [
            [0.5, 0.2], 
            [0.6, -0.6]
        ], 
        [
            [0.8, 0.4], 
        ],
    ],
    [
        [
            [0.3], 
            [0.25]
        ], 
        [
            [-0.5]
        ]
    ],
    loss_function = 'binary_cross_entropy',
    output_activation_function=activation_function["Sigmoid"],
)

network_expected_outputs = [1]
network_outputs = 0
for i in range(1000):
    network_outputs = network_2_2.forward([0.75, 1.25])
    network_2_2.backward(network_expected_outputs)
    network_2_2.zero_grad(0.1)

print('Total Loss',network_2_2.binary_cross_entropy(network_outputs, network_expected_outputs))