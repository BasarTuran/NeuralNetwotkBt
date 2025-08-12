namespace NeuralNetworkBT;

using System;

using System;
using System.Linq;

public static class ActivationFunctions
{
    public static Func<double, double> Get(string name) => name switch
    {
        "Sigmoid" => Sigmoid,
        "ReLU" => ReLU,
        "Tanh" => Tanh,
        "LeakyReLU" => LeakyReLU,
        "ELU" => ELU,
        "Swish" => Swish,
        _ => Sigmoid,
    };

    public static Func<double, double> GetDerivative(string name) => name switch
    {
        "Sigmoid" => SigmoidDerivative,
        "ReLU" => ReLUDerivative,
        "Tanh" => TanhDerivative,
        "LeakyReLU" => LeakyReLUDerivative,
        "ELU" => ELUDerivative,
        "Swish" => SwishDerivative,
        _ => SigmoidDerivative,
    };

    public static Func<double[], double[]> GetVector(string name) => name switch
    {
        "Softmax" => Softmax,
        _ => null
    };

    private static double Sigmoid(double x) => 1.0 / (1.0 + Math.Exp(-x));
    private static double SigmoidDerivative(double y) => y * (1 - y);

    private static double ReLU(double x) => x > 0 ? x : 0;
    private static double ReLUDerivative(double y) => y > 0 ? 1 : 0;

    private static double Tanh(double x) => Math.Tanh(x);
    private static double TanhDerivative(double y) => 1 - y * y;

    private static double LeakyReLU(double x) => x > 0 ? x : 0.01 * x;
    private static double LeakyReLUDerivative(double y) => y > 0 ? 1 : 0.01;

    private static double ELU(double x) => x >= 0 ? x : (Math.Exp(x) - 1);
    private static double ELUDerivative(double y) => y >= 0 ? 1 : y + 1;

    private static double Swish(double x) => x / (1 + Math.Exp(-x));
    private static double SwishDerivative(double y) => y + Sigmoid(y) * (1 - y);

    public static double[] Softmax(double[] values)
    {
        double max = values.Max();
        double[] exp = values.Select(v => Math.Exp(v - max)).ToArray();
        double sum = exp.Sum();
        return exp.Select(e => e / sum).ToArray();
    }
}
