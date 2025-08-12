namespace NeuralNetworkBT;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

public class NeuralNetwork
{
    private readonly NeuralNetworkConfig _config;
    private readonly Func<double, double> _activation;
    private readonly Func<double, double> _activationDerivative;
    private readonly Func<double[], double[]> _activationVector; // Softmax gibi
    private readonly Func<double[], double[], double> _lossFunction;
    private readonly Func<double[], double[], double[]> _lossDerivative;

    private List<double[,]> Weights;
    private List<double[]> Biases;

    private List<double[,]> mWeights;
    private List<double[,]> vWeights;
    private List<double[]> mBiases;
    private List<double[]> vBiases;

    private int t; // Adam için timestep
    public List<double[,]> GetWeights()
    {
        return this.Weights; // weights internal alanınız
    }

    public List<double[]> GetBiases()
    {
        return this.Biases; // biases internal alanınız
    }
    public NeuralNetwork(NeuralNetworkConfig config)
    {
        _config = config;

        _activation = ActivationFunctions.Get(config.Activation);
        _activationDerivative = ActivationFunctions.GetDerivative(config.Activation);
        _activationVector = ActivationFunctions.GetVector(config.Activation);

        if (config.LossFunction == "CrossEntropy")
        {
            _lossFunction = LossFunctions.CrossEntropyLoss;
            _lossDerivative = LossFunctions.CrossEntropyLossDerivative;
        }
        else
        {
            _lossFunction = LossFunctions.MeanSquaredError;
            _lossDerivative = LossFunctions.MeanSquaredErrorDerivative;
        }

        InitializeNetwork();
        InitializeAdam();
    }

    private void InitializeNetwork()
    {
        Weights = new List<double[,]>();
        Biases = new List<double[]>();

        var layerSizes = new List<int> { _config.Input.Count };
        layerSizes.AddRange(_config.HiddenLayers.Select(h => h.NeuronCount));
        layerSizes.Add(_config.Output.Count);

        var rand = new Random();
        for (int i = 0; i < layerSizes.Count - 1; i++)
        {
            var w = new double[layerSizes[i + 1], layerSizes[i]];
            var b = new double[layerSizes[i + 1]];

            for (int r = 0; r < w.GetLength(0); r++)
            {
                for (int c = 0; c < w.GetLength(1); c++)
                    w[r, c] = rand.NextDouble() - 0.5;

                b[r] = _config.Bias ? rand.NextDouble() - 0.5 : 0.0;
            }

            Weights.Add(w);
            Biases.Add(b);
        }
    }

    private void InitializeAdam()
    {
        mWeights = Weights.Select(w => new double[w.GetLength(0), w.GetLength(1)]).ToList();
        vWeights = Weights.Select(w => new double[w.GetLength(0), w.GetLength(1)]).ToList();
        mBiases = Biases.Select(b => new double[b.Length]).ToList();
        vBiases = Biases.Select(b => new double[b.Length]).ToList();
        t = 0;
    }

    public double[] Forward(double[] input)
    {
        return ForwardInternal(input).Last();
    }

    private List<double[]> ForwardInternal(double[] input)
    {
        var outputs = new List<double[]> { input };
        double[] currentOutput = input;

        for (int layer = 0; layer < Weights.Count; layer++)
        {
            int neuronCount = Weights[layer].GetLength(0);
            int prevLayerCount = Weights[layer].GetLength(1);
            double[] nextLayerOutput = new double[neuronCount];

            Parallel.For(0, neuronCount, neuron =>
            {
                double sum = 0;
                for (int w = 0; w < prevLayerCount; w++)
                {
                    sum += Weights[layer][neuron, w] * currentOutput[w];
                }
                sum += Biases[layer][neuron];
                nextLayerOutput[neuron] = _activation(sum);
            });

            if (layer == Weights.Count - 1 && _activationVector != null)
            {
                nextLayerOutput = _activationVector(nextLayerOutput);
            }

            outputs.Add(nextLayerOutput);
            currentOutput = nextLayerOutput;
        }

        return outputs;
    }

    private List<double[]> Backpropagate(List<double[]> layerOutputs, double[] target)
    {
        int L = Weights.Count;
        var deltas = new List<double[]>(L);
        for (int i = 0; i < L; i++) deltas.Add(null);

        var output = layerOutputs.Last();

        if (_config.LossFunction == "CrossEntropy" && _activationVector != null)
        {
            double[] deltaOutput = new double[output.Length];
            Parallel.For(0, output.Length, i =>
            {
                deltaOutput[i] = output[i] - target[i];
            });
            deltas[L - 1] = deltaOutput;
        }
        else
        {
            var lossGrad = _lossDerivative(output, target);
            double[] deltaOutput = new double[output.Length];
            Parallel.For(0, output.Length, i =>
            {
                deltaOutput[i] = lossGrad[i] * _activationDerivative(output[i]);
            });
            deltas[L - 1] = deltaOutput;
        }

        for (int layer = L - 2; layer >= 0; layer--)
        {
            int size = Weights[layer].GetLength(0); // nöron sayısı
            double[] delta = new double[size];

            Parallel.For(0, size, i =>
            {
                double error = 0;
                int nextLayerNeurons = Weights[layer + 1].GetLength(0);
                for (int j = 0; j < nextLayerNeurons; j++)
                {
                    error += Weights[layer + 1][j, i] * deltas[layer + 1][j];
                }
                delta[i] = error * _activationDerivative(layerOutputs[layer + 1][i]);
            });

            deltas[layer] = delta;
        }
        return deltas;
    }

    private void UpdateWeights(List<double[]> layerOutputs, List<double[]> deltas)
    {
        t++;

        double beta1 = 0.9;
        double beta2 = 0.999;
        double epsilon = 1e-8;

        for (int layer = 0; layer < Weights.Count; layer++)
        {
            int neuronCount = Weights[layer].GetLength(0);
            int weightCount = Weights[layer].GetLength(1);

            Parallel.For(0, neuronCount, neuron =>
            {
                for (int w = 0; w < weightCount; w++)
                {
                    double grad = deltas[layer][neuron] * layerOutputs[layer][w];

                    mWeights[layer][neuron, w] = beta1 * mWeights[layer][neuron, w] + (1 - beta1) * grad;
                    vWeights[layer][neuron, w] = beta2 * vWeights[layer][neuron, w] + (1 - beta2) * grad * grad;

                    double mHat = mWeights[layer][neuron, w] / (1 - Math.Pow(beta1, t));
                    double vHat = vWeights[layer][neuron, w] / (1 - Math.Pow(beta2, t));

                    Weights[layer][neuron, w] -= _config.LearningRate * mHat / (Math.Sqrt(vHat) + epsilon);
                }

                double gradBias = deltas[layer][neuron];

                mBiases[layer][neuron] = beta1 * mBiases[layer][neuron] + (1 - beta1) * gradBias;
                vBiases[layer][neuron] = beta2 * vBiases[layer][neuron] + (1 - beta2) * gradBias * gradBias;

                double mHatBias = mBiases[layer][neuron] / (1 - Math.Pow(beta1, t));
                double vHatBias = vBiases[layer][neuron] / (1 - Math.Pow(beta2, t));

                Biases[layer][neuron] -= _config.LearningRate * mHatBias / (Math.Sqrt(vHatBias) + epsilon);
            });
        }
    }

    public void Train(double[][] inputs, double[][] targets)
    {
        int noImprovementEpochs = 0;
        double bestLoss = double.MaxValue;

        for (int epoch = 0; epoch < _config.Epochs; epoch++)
        {
            double totalLoss = 0;

            for (int batchStart = 0; batchStart < inputs.Length; batchStart += _config.BatchSize)
            {
                int batchEnd = Math.Min(batchStart + _config.BatchSize, inputs.Length);
                var batchInputs = inputs[batchStart..batchEnd];
                var batchTargets = targets[batchStart..batchEnd];

                var batchGradients = new List<(List<double[]> outputs, List<double[]> deltas)>();

                for (int i = 0; i < batchInputs.Length; i++)
                {
                    var outputs = ForwardInternal(batchInputs[i]);
                    var deltas = Backpropagate(outputs, batchTargets[i]);
                    batchGradients.Add((outputs, deltas));
                    totalLoss += _lossFunction(outputs.Last(), batchTargets[i]);
                }

                foreach (var (outputs, deltas) in batchGradients)
                {
                    UpdateWeights(outputs, deltas);
                }
            }

            totalLoss /= inputs.Length;
            if(epoch%1000 == 0)
                Console.WriteLine($"Epoch {epoch }/{_config.Epochs} - Loss: {totalLoss:F6}");

            if (totalLoss < bestLoss)
            {
                bestLoss = totalLoss;
                noImprovementEpochs = 0;
                ModelStorage.Save(_config.ModelFile, Weights, Biases);
            }
            else
            {
                noImprovementEpochs++;
                if (noImprovementEpochs >= _config.EarlyStoppingPatience && _config.EarlyStoppingPatience>0)
                {
                    Console.WriteLine("Early stopping triggered."+_config.EarlyStoppingPatience);
                    break;
                }
            }
        }
    }

    public double Evaluate(double[][] inputs, double[][] targets)
    {
        double loss = 0;
        for (int i = 0; i < inputs.Length; i++)
        {
            var pred = Forward(inputs[i]);
            loss += _lossFunction(pred, targets[i]);
        }
        return loss / inputs.Length;
    }

    public void LoadModel(string path)
    {
        (Weights, Biases) = ModelStorage.Load(path);
        InitializeAdam();
    }
}
