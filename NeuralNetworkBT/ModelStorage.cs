

namespace NeuralNetworkBT;
using System.Collections.Generic;
using System.IO;
using System.Text.Json;

public static class ModelStorage
{
    public static void Save(string path, List<double[,]> weights, List<double[]> biases)
    {
        var model = new
        {
            Weights = weights.Select(matrix => ConvertToJagged(matrix)).ToList(),
            Biases = biases
        };
        var options = new JsonSerializerOptions { WriteIndented = true };
        string json = JsonSerializer.Serialize(model, options);
        File.WriteAllText(path, json);
    }

    public static (List<double[,]>, List<double[]>) Load(string path)
    {
        string json = File.ReadAllText(path);
        var doc = JsonDocument.Parse(json);
        var weights = new List<double[,]>();
        var biases = new List<double[]>();

        var root = doc.RootElement;

        var weightsElem = root.GetProperty("Weights");
        foreach (var wElem in weightsElem.EnumerateArray())
        {
            var rows = wElem.GetArrayLength();
            var cols = wElem[0].GetArrayLength();
            double[,] w = new double[rows, cols];
            int r = 0;
            foreach (var rowElem in wElem.EnumerateArray())
            {
                int c = 0;
                foreach (var val in rowElem.EnumerateArray())
                {
                    w[r, c++] = val.GetDouble();
                }
                r++;
            }
            weights.Add(w);
        }

        var biasesElem = root.GetProperty("Biases");
        foreach (var bElem in biasesElem.EnumerateArray())
        {
            double[] b = new double[bElem.GetArrayLength()];
            int i = 0;
            foreach (var val in bElem.EnumerateArray())
            {
                b[i++] = val.GetDouble();
            }
            biases.Add(b);
        }

        return (weights, biases);
    }
    public static List<double[]> ConvertToJagged(double[,] array2D)
    {
        int rows = array2D.GetLength(0);
        int cols = array2D.GetLength(1);
        var jagged = new List<double[]>(rows);
        for (int i = 0; i < rows; i++)
        {
            var row = new double[cols];
            for (int j = 0; j < cols; j++)
            {
                row[j] = array2D[i, j];
            }
            jagged.Add(row);
        }
        return jagged;
    }
}
