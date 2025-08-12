namespace NeuralNetworkBT;

using System;

public static class LossFunctions
{
    public static double MeanSquaredError(double[] predicted, double[] actual)
    {
        double sum = 0;
        for (int i = 0; i < predicted.Length; i++)
        {
            double diff = actual[i] - predicted[i];
            sum += diff * diff;
        }
        return sum / predicted.Length;
    }

    public static double[] MeanSquaredErrorDerivative(double[] predicted, double[] actual)
    {
        double[] grad = new double[predicted.Length];
        for (int i = 0; i < predicted.Length; i++)
        {
            grad[i] = 2 * (predicted[i] - actual[i]) / predicted.Length;
        }
        return grad;
    }

    public static double CrossEntropyLoss(double[] predicted, double[] actual)
    {
        double epsilon = 1e-15;
        double loss = 0;
        for (int i = 0; i < predicted.Length; i++)
        {
            double p = Math.Min(Math.Max(predicted[i], epsilon), 1 - epsilon);
            loss -= actual[i] * Math.Log(p);
        }
        return loss / predicted.Length;
    }

    public static double[] CrossEntropyLossDerivative(double[] predicted, double[] actual)
    {
        double epsilon = 1e-15;
        double[] grad = new double[predicted.Length];
        for (int i = 0; i < predicted.Length; i++)
        {
            double p = Math.Min(Math.Max(predicted[i], epsilon), 1 - epsilon);
            grad[i] = -(actual[i] / p);
        }
        return grad;
    }
}
