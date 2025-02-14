import math
import copy
# from decimal import Decimal, getcontext

# getcontext().prec = 28
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
        self.input_array = []


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
    # def mean_squared_error(self, outputs, expected_outputs):
    #     # 使用 Decimal 确保每一个数值的计算都在高精度范围内
    #     mse = sum(
    #         (Decimal(expect) - Decimal(output)) ** 2 
    #         for output, expect in zip(outputs, expected_outputs)
    #     ) / Decimal(len(outputs))
    #     self.loss_function = 'mean_squared_error'
    #     return mse

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
        self.input_array = inputs
        layer_len = len(self.weights)
        self.output_array = []
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
            [{'output_gradient': 0, 'net_output_gradient': 0} for _ in range(len(self.output_array[layer]))]
            for layer in range(layer_len)
        ]

        # 紀錄每一層的輸出值
        for layer in range(layer_len - 1, -1, -1):
            for column in range(len(self.output_array[layer])):
                loss_gradient = 0
                output_gradient = 0
                if layer == layer_len - 1:
                    # 計算輸出層的損失函數 mse 的 derivative
                    if self.loss_function == 'mean_squared_error':
                        loss_gradient = (2 / len(self.output_array[layer])) * (self.output_array[layer][column] - expected_outputs[column])
                    elif self.loss_function == 'binary_cross_entropy':
                        expect = expected_outputs[column]
                        output = self.output_array[layer][column]
                        # 被修正的地方
                        # output = min(max(output, 1e-15), 1 - 1e-15)  # 防止 log(0)
                        output = self.output_array[layer][column]
                        loss_gradient = -((expect / output) - (1 - expect) / (1 - output))

                    output_gradient = loss_gradient
                    self.neuron_gradient[layer][column]['output_gradient'] = output_gradient

                # else:
                #     # 计算隐藏层的梯度
                #     for next_layer_column in range(len(self.weights[layer + 1])):
                #         gradient_contribution = (
                #             self.weights[layer + 1][next_layer_column][column]  * self.neuron_gradient[layer + 1][next_layer_column]['net_output_gradient']
                #         )
                #         self.neuron_gradient[layer][column]['output_gradient'] += gradient_contribution

                elif column == 0:
                        # 要將第二次column的跟第一次column的相加
                        for array in range(len(self.weights[layer+1])):
                            for weight in range(len(self.weights[layer+1][array])):
                                # output_gradient = self.weights[layer+1][array][weight] * self.output_array[layer+1][array]
                                output_gradient = self.weights[layer+1][array][weight] * self.neuron_gradient[layer+1][array]['net_output_gradient']
                                self.neuron_gradient[layer][weight]['output_gradient'] += output_gradient

                # 計算激活函數的 derivative
                net_output_gradient = 0
                if layer == layer_len - 1:
                    if self.output_activation_function == activation_function['Linear']:
                        net_output_gradient = 1
                    elif self.output_activation_function == activation_function['Sigmoid']:
                        output = self._Sigmoid(self.output_array[layer][column])
                        net_output_gradient = output * (1 - output)

                elif self.other_activation_function.get(layer, None):
                    if self.other_activation_function[layer] == activation_function['Linear']:
                        net_output_gradient = 1
                    elif self.other_activation_function[layer] == activation_function['Sigmoid']:
                        output = self.output_array[layer][column]
                        net_output_gradient = output * (1 - output)
                else:  # Default to ReLU for hidden layers
                    # 被修正的地方
                    if self.output_array[layer][column] > 0:
                        net_output_gradient = 1
                    else:
                        net_output_gradient = 0

                self.neuron_gradient[layer][column]['net_output_gradient'] = net_output_gradient
                # self.neuron_gradient[layer][column]['net_output_gradient'] = (
                #     self.neuron_gradient[layer][column]['output_gradient'] * net_output_gradient
                # )

                # 計算權重的 gradient
                if layer > 0:
                    for output in range(len(self.output_array[layer - 1])):
                        weight_gradient = (
                            self.neuron_gradient[layer][column]['net_output_gradient']
                            * self.output_array[layer - 1][output]
                        )
                        self.new_weights_gradient[layer][column][output] = weight_gradient
                else:
                    for input_idx in range(len(self.input_array)):
                        weight_gradient = (
                            self.neuron_gradient[layer][column]['net_output_gradient']
                            * self.input_array[input_idx]
                        )
                        self.new_weights_gradient[layer][column][input_idx] = weight_gradient

                # 計算bias的 gradient
                self.new_biasesWeights_gradient[layer][column][0] = (
                    self.neuron_gradient[layer][column]['net_output_gradient']
                )

    def zero_grad(self, learning_rate):
        for layer in range(len(self.new_weights_gradient)):
            for array in range(len(self.new_weights_gradient[layer])):
                for column in range(len(self.new_weights_gradient[layer][array])):
                    old_weight = self.weights[layer][array][column]
                    # gradient = self.new_weights_gradient[layer][array][column]
                    gradient = self.new_weights_gradient[layer][array][column] * self.neuron_gradient[layer][array]['net_output_gradient'] * self.neuron_gradient[layer][array]['output_gradient']
                    # 權重更新
                    self.weights[layer][array][column] = old_weight - learning_rate * gradient

                bias_old_weight = self.biasesWeights[layer][array][0]
                bias_gradient =  self.new_biasesWeights_gradient[layer][array][0] * self.neuron_gradient[layer][array]['net_output_gradient'] * self.neuron_gradient[layer][array]['output_gradient']
                # bias_gradient = self.new_biasesWeights_gradient[layer][array][0]
                self.biasesWeights[layer][array][0] = bias_old_weight - learning_rate * bias_gradient
                    

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
print("Outputs", network_outputs)
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
