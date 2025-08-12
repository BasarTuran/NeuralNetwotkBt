namespace NeuralNetworkBT;

using System;
using System.Linq;

public static class NormalizationFunctions
{
    public static double[] MinMax(double[] values)
    {
        double min = values.Min();
        double max = values.Max();
        if (max == min) return values.Select(_ => 0.5).ToArray();
        return values.Select(v => (v - min) / (max - min)).ToArray();
    }

    public static double[] ZScore(double[] values)
    {
        double mean = values.Average();
        double stdDev = Math.Sqrt(values.Sum(v => Math.Pow(v - mean, 2)) / values.Length);
        if (stdDev == 0) return values.Select(v => 0.0).ToArray();
        return values.Select(v => (v - mean) / stdDev).ToArray();
    }

    public static double[] DecimalScaling(double[] values)
    {
        double maxAbs = values.Select(v => Math.Abs(v)).Max();
        int j = (int)Math.Ceiling(Math.Log10(maxAbs + 1));
        if (j == 0) return values;
        return values.Select(v => v / Math.Pow(10, j)).ToArray();
    }

    public static double[] MeanNormalization(double[] values)
    {
        double mean = values.Average();
        double min = values.Min();
        double max = values.Max();
        if (max - min == 0) return values.Select(v => 0.0).ToArray();
        return values.Select(v => (v - mean) / (max - min)).ToArray();
    }

    public static double[] L2Normalize(double[] values)
    {
        double norm = Math.Sqrt(values.Sum(v => v * v));
        if (norm == 0) return values;
        return values.Select(v => v / norm).ToArray();
    }

    public static double[] RobustScale(double[] values)
    {
        double[] sorted = values.OrderBy(v => v).ToArray();
        int n = sorted.Length;

        double median = Median(sorted);

        double q1 = Median(sorted.Take(n / 2).ToArray());
        double q3 = Median(sorted.Skip((n + 1) / 2).ToArray());

        double iqr = q3 - q1;
        if (iqr == 0) return values;

        return values.Select(v => (v - median) / iqr).ToArray();
    }

    private static double Median(double[] arr)
    {
        int n = arr.Length;
        if (n == 0) return 0;
        if (n % 2 == 0)
            return (arr[n / 2 - 1] + arr[n / 2]) / 2.0;
        else
            return arr[n / 2];
    }
}
