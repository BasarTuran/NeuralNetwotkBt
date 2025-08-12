namespace NeuralNetworkBT;

using System;
using System.IO;
using System.Linq;
using System.Text.Json;
using NeuralNetworkBT;

public static class NeuralNetworkRunner
{
    public class TrainingData
    {
        public double[] Input { get; set; }
        public double[] Output { get; set; }
    }

    private static double[] ApplyMinMax(double[] v, double[] min, double[] max)
    {
        var y = new double[v.Length];
        for (int i = 0; i < v.Length; i++)
        {
            double range = Math.Max(1e-12, max[i] - min[i]);
            y[i] = (v[i] - min[i]) / range;
        }
        return y;
    }

    private static double[] InverseMinMax(double[] vn, double[] min, double[] max)
    {
        var y = new double[vn.Length];
        for (int i = 0; i < vn.Length; i++)
        {
            double range = Math.Max(1e-12, max[i] - min[i]);
            y[i] = vn[i] * range + min[i];
        }
        return y;
    }

    public static void Train(string configPath)
    {
        var configJson = File.ReadAllText(configPath);
        var config = JsonSerializer.Deserialize<NeuralNetworkConfig>(configJson);

        var nn = new NeuralNetwork(config);

        var trainingJson = File.ReadAllText(config.TrainingDataFile);
        var trainingData = JsonSerializer.Deserialize<TrainingData[]>(trainingJson);

        int inDim = config.Input.Count;
        int outDim = config.Output.Count;

        // Min/Max hesapla
        var inMin  = Enumerable.Repeat(double.PositiveInfinity, inDim).ToArray();
        var inMax  = Enumerable.Repeat(double.NegativeInfinity, inDim).ToArray();
        var outMin = Enumerable.Repeat(double.PositiveInfinity, outDim).ToArray();
        var outMax = Enumerable.Repeat(double.NegativeInfinity, outDim).ToArray();

        foreach (var td in trainingData)
        {
            for (int i = 0; i < inDim; i++)
            {
                inMin[i] = Math.Min(inMin[i], td.Input[i]);
                inMax[i] = Math.Max(inMax[i], td.Input[i]);
            }
            for (int j = 0; j < outDim; j++)
            {
                outMin[j] = Math.Min(outMin[j], td.Output[j]);
                outMax[j] = Math.Max(outMax[j], td.Output[j]);
            }
        }

        // Normalize et
        double[][] inputs  = trainingData.Select(td => ApplyMinMax(td.Input, inMin, inMax)).ToArray();
        double[][] outputs = trainingData.Select(td => ApplyMinMax(td.Output, outMin, outMax)).ToArray();

        // Eğit
        nn.Train(inputs, outputs);

        // Kaydet
        ModelStorage.SaveWithNormalization(
            config.ModelFile,
            nn.GetWeights(),
            nn.GetBiases(),
            new ModelStorage.NormalizationParams
            {
                InMin = inMin,
                InMax = inMax,
                OutMin = outMin,
                OutMax = outMax
            }
        );

        Console.WriteLine("Eğitim tamamlandı ve model kaydedildi.");
    }

    public static void Predict(string configPath)
    {
        var configJson = File.ReadAllText(configPath);
        var config = JsonSerializer.Deserialize<NeuralNetworkConfig>(configJson);

        var nn = new NeuralNetwork(config);

        if (!File.Exists(config.ModelFile))
        {
            Console.WriteLine("Model dosyası bulunamadı. Önce eğitim yapın.");
            return;
        }

        // Ağırlıkları yükle
        nn.LoadModel(config.ModelFile);

        // Normalizasyon parametrelerini oku
        var modelText = File.ReadAllText(config.ModelFile);
        var modelDto = JsonSerializer.Deserialize<ModelDto>(modelText);
        if (modelDto?.Normalization == null)
        {
            Console.WriteLine("Normalizasyon parametreleri bulunamadı.");
            return;
        }

        var inMin  = modelDto.Normalization.InMin;
        var inMax  = modelDto.Normalization.InMax;
        var outMin = modelDto.Normalization.OutMin;
        var outMax = modelDto.Normalization.OutMax;

        Console.WriteLine($"Tahmin için giriş verisini giriniz ({config.Input.Count} adet double):");
        string inputLine = Console.ReadLine();
        var inputStrs = inputLine.Split(' ', StringSplitOptions.RemoveEmptyEntries);

        if (inputStrs.Length != config.Input.Count)
        {
            Console.WriteLine("Hatalı giriş sayısı.");
            return;
        }

        var inputValues = new double[config.Input.Count];
        for (int i = 0; i < config.Input.Count; i++)
        {
            if (!double.TryParse(inputStrs[i], out inputValues[i]))
            {
                Console.WriteLine("Geçersiz sayı.");
                return;
            }
        }

        var normInput = ApplyMinMax(inputValues, inMin, inMax);
        var normOutput = nn.Forward(normInput);
        var denormOutput = InverseMinMax(normOutput, outMin, outMax);

        Console.WriteLine("Tahmin sonucu:");
        for (int i = 0; i < denormOutput.Length; i++)
            Console.WriteLine($"Output[{i}]: {denormOutput[i]:F4}");
    }

    private class ModelDto
    {
        public System.Collections.Generic.List<double[][]> Weights { get; set; }
        public System.Collections.Generic.List<double[]> Biases { get; set; }
        public ModelStorage.NormalizationParams Normalization { get; set; }
    }
}
