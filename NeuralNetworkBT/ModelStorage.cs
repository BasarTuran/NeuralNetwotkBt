using System;
using System.Collections.Generic;
using System.IO;
using System.Text.Json;

public static class ModelStorage
{
    // Yardımcı: 2D -> jagged
    private static double[][] ToJagged(double[,] m)
    {
        int r = m.GetLength(0), c = m.GetLength(1);
        var j = new double[r][];
        for (int i = 0; i < r; i++)
        {
            j[i] = new double[c];
            for (int k = 0; k < c; k++) j[i][k] = m[i, k];
        }
        return j;
    }

    // Yardımcı: jagged -> 2D
    private static double[,] ToRect(double[][] j)
    {
        int r = j.Length, c = j[0].Length;
        var m = new double[r, c];
        for (int i = 0; i < r; i++)
            for (int k = 0; k < c; k++)
                m[i, k] = j[i][k];
        return m;
    }

    public class NormalizationParams
    {
        public double[] InMin  { get; set; }
        public double[] InMax  { get; set; }
        public double[] OutMin { get; set; }
        public double[] OutMax { get; set; }
    }

    private class ModelDto
    {
        public List<double[][]> Weights { get; set; }
        public List<double[]>   Biases  { get; set; }
        public NormalizationParams Normalization { get; set; }
    }

    // Eski imza: sadece ağırlık + bias (geriye uyum için kalsın)
    public static void Save(string path, List<double[,]> weights, List<double[]> biases)
    {
        var dto = new ModelDto
        {
            Weights = weights.ConvertAll(ToJagged),
            Biases  = biases
        };
        var options = new JsonSerializerOptions { WriteIndented = true };
        File.WriteAllText(path, JsonSerializer.Serialize(dto, options));
    }

    // YENİ: normalizasyonla birlikte kaydet
    public static void SaveWithNormalization(
        string path,
        List<double[,]> weights,
        List<double[]> biases,
        NormalizationParams norm)
    {
        var dto = new ModelDto
        {
            Weights = weights.ConvertAll(ToJagged),
            Biases  = biases,
            Normalization = norm
        };
        var options = new JsonSerializerOptions { WriteIndented = true };
        File.WriteAllText(path, JsonSerializer.Serialize(dto, options));
    }

    // Eski Load: (ağ içinde kullanılıyor)
    public static (List<double[,]> Weights, List<double[]> Biases) Load(string path)
    {
        var text = File.ReadAllText(path);
        var dto  = JsonSerializer.Deserialize<ModelDto>(text);
        var w    = new List<double[,]>();
        foreach (var j in dto.Weights) w.Add(ToRect(j));
        return (w, dto.Biases);
    }

    // İstersen normalizasyonu da birlikte döndüren ek Load:
    public static (List<double[,]> Weights, List<double[]> Biases, NormalizationParams Norm) LoadWithNormalization(string path)
    {
        var text = File.ReadAllText(path);
        var dto  = JsonSerializer.Deserialize<ModelDto>(text);
        var w    = new List<double[,]>();
        foreach (var j in dto.Weights) w.Add(ToRect(j));
        return (w, dto.Biases, dto.Normalization);
    }

    // Debug için (sende zaten var gibi)
    public static void PrintWeights(List<double[,]> weights)
    {
        for (int l = 0; l < weights.Count; l++)
        {
            var w = weights[l];
            Console.WriteLine($"Layer {l}:");
            int r = w.GetLength(0), c = w.GetLength(1);
            for (int i = 0; i < r; i++)
            {
                Console.Write("  Neuron " + i + ": ");
                for (int j = 0; j < c; j++)
                    Console.Write($"{w[i, j]:F4} ");
                Console.WriteLine();
            }
        }
    }
}
