# Define the training data XOR
$train = @(
    @(0, 0, 0),
    @(0, 1, 1),
    @(1, 0, 1),
    @(1, 1, 0)
)

# Define the neural network parameters
$num_inputs = 3
$num_hidden_neurons = 4
$num_outputs = 2
$learning_rate = 1
$num_iterations = 10000

# Declare the weight arrays
$hidden_weights = [float[,]]::new($num_inputs, $num_hidden_neurons)
$output_weights = [float[,]]::new($num_hidden_neurons, $num_outputs)

# Generate a random float between -1 and 1
function rand_float() {
    return (Get-Random) / [System.Int32]::MaxValue * 2.0 - 1.0
}

# Activation function (sigmoid)
function sigmoid($x) {
    return 1.0 / (1.0 + [Math]::Exp(-$x))
}

# Calculate the cost function
function cost($y_pred) {
    $result = 0.0
    $n = $train.Count

    for ($i = 0; $i -lt $n; $i++) {
        $y = $train[$i][$num_inputs]
        $d = $y - $y_pred[$i]
        $result += $d * $d
    }

    $result /= $n
    return $result
}

# Forward pass through the neural network
function forward_pass($inputs, $hidden_neurons, $y_pred) {
    # Calculate the activations of the hidden neurons
    for ($j = 0; $j -lt $num_hidden_neurons; $j++) {
        $activation = 0.0
        for ($i = 0; $i -lt $num_inputs; $i++) {
            $activation += $inputs[$i] * $hidden_weights[$i, $j]
        }
        $hidden_neurons[$j] = sigmoid($activation)
    }

    # Calculate the output prediction
    for ($k = 0; $k -lt $num_outputs; $k++) {
        $activation = 0.0
        for ($j = 0; $j -lt $num_hidden_neurons; $j++) {
            $activation += $hidden_neurons[$j] * $output_weights[$j, $k]
        }
        $y_pred[$k] = sigmoid($activation)
    }
}

# Backpropagation algorithm to update weights
function backpropagation($inputs, $hidden_neurons, $y_pred, $targets) {
    # Calculate the error term for the output layer
    $output_delta = [float[]]::new($num_outputs)
    for ($k = 0; $k -lt $num_outputs; $k++) {
        $err = $targets[$k] - $y_pred[$k]
        $output_delta[$k] = $err * $y_pred[$k] * (1.0 - $y_pred[$k])
    }

    # Update the weights between hidden layer and output layer
    for ($j = 0; $j -lt $num_hidden_neurons; $j++) {
        for ($k = 0; $k -lt $num_outputs; $k++) {
            $output_weights[$j, $k] += $learning_rate * $hidden_neurons[$j] * $output_delta[$k]
        }
    }

    # Calculate the error term for the hidden layer
    $hidden_delta = [float[]]::new($num_hidden_neurons)
    for ($j = 0; $j -lt $num_hidden_neurons; $j++) {
        $err = 0.0
        for ($k = 0; $k -lt $num_outputs; $k++) {
            $err += $output_delta[$k] * $output_weights[$j, $k]
        }
        $hidden_delta[$j] = $err * $hidden_neurons[$j] * (1.0 - $hidden_neurons[$j])
    }

    # Update the weights between input layer and hidden layer
    for ($i = 0; $i -lt $num_inputs; $i++) {
        for ($j = 0; $j -lt $num_hidden_neurons; $j++) {
            $hidden_weights[$i, $j] += $learning_rate * $inputs[$i] * $hidden_delta[$j]
        }
    }
}

# Set the random seed
[void][Random]::new().Next()

# Initialize the weights with random values
for ($i = 0; $i -lt $num_inputs; $i++) {
    for ($j = 0; $j -lt $num_hidden_neurons; $j++) {
        $hidden_weights[$i, $j] = rand_float
    }
}
for ($j = 0; $j -lt $num_hidden_neurons; $j++) {
    for ($k = 0; $k -lt $num_outputs; $k++) {
        $output_weights[$j, $k] = rand_float
    }
}

# Train the neural network
for ($iteration = 0; $iteration -lt $num_iterations; $iteration++) {
    # Select a random training example
    $index = Get-Random -Minimum 0 -Maximum $train.Count
    $inputs = $train[$index]
    $targets = $train[$index][$num_inputs..($train[$index].Length - 1)]

    # Perform a forward pass
    $hidden_neurons = [float[]]::new($num_hidden_neurons)
    $y_pred = [float[]]::new($num_outputs)
    forward_pass $inputs $hidden_neurons $y_pred

    # Perform backpropagation to update weights
    backpropagation $inputs $hidden_neurons $y_pred $targets
}

# Print the final predictions for the training data
Write-Host "Training Data Predictions:"
for ($i = 0; $i -lt $train.Count; $i++) {
    $inputs = $train[$i]
    $hidden_neurons = [float[]]::new($num_hidden_neurons)
    $y_pred = [float[]]::new($num_outputs)

    forward_pass $inputs $hidden_neurons $y_pred
    Write-Host "Input: $($inputs[0]), $($inputs[1]) => Trained Output: $($inputs[2]) Predicted Output: (Rounded) $([math]::round($y_pred[0])) | (Unrounded) $($y_pred[0])"
}
pause