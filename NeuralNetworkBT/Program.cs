// See https://aka.ms/new-console-template for more information

using System.Text.Json;
using NeuralNetworkBT;

static void Main()
    {
        Console.WriteLine("Yapay Sinir Ağı Uygulamasına Hoşgeldiniz.");
        Console.WriteLine("1 - Eğitim Yap");
        Console.WriteLine("2 - Tahmin Yap");
        Console.Write("Seçiminiz (1/2): ");

        var choice = Console.ReadLine();

        string configPath = "appsettings.json";
        var configJson = File.ReadAllText(configPath);
        var config = JsonSerializer.Deserialize<NeuralNetworkConfig>(configJson);

        var nn = new NeuralNetwork(config);

        Func<double[], double[]> normalizer = config.Normalization switch
        {
            "MinMax" => NormalizationFunctions.MinMax,
            "ZScore" => NormalizationFunctions.ZScore,
            "DecimalScaling" => NormalizationFunctions.DecimalScaling,
            "MeanNormalization" => NormalizationFunctions.MeanNormalization,
            "L2Normalize" => NormalizationFunctions.L2Normalize,
            "RobustScale" => NormalizationFunctions.RobustScale,
            _ => (double[] arr) => arr // normalization yok
        };

        if (choice == "1")
        {
            var trainingJson = File.ReadAllText(config.TrainingDataFile);
            var trainingData = JsonSerializer.Deserialize<TrainingData[]>(trainingJson);

            double[][] inputs = trainingData.Select(td => normalizer(td.Input)).ToArray();
            double[][] outputs = trainingData.Select(td => td.Output).ToArray();

            nn.Train(inputs, outputs);
            ModelStorage.Save(config.ModelFile,nn.GetWeights(),nn.GetBiases());
            Console.WriteLine("Eğitim tamamlandı.");
        }
        else if (choice == "2")
        {
            if (!File.Exists(config.ModelFile))
            {
                Console.WriteLine("Model dosyası bulunamadı. Önce eğitim yapmalısınız.");
                return;
            }

            nn.LoadModel(config.ModelFile);

            Console.WriteLine($"Tahmin için giriş verisini giriniz. ({config.Input.Count} adet double, aralarda boşluk)");

            string inputLine = Console.ReadLine();
            var inputStrs = inputLine.Split(' ', StringSplitOptions.RemoveEmptyEntries);

            if (inputStrs.Length != config.Input.Count)
            {
                Console.WriteLine($"Hatalı giriş. {config.Input.Count} değer girmelisiniz.");
                return;
            }

            double[] inputValues = new double[config.Input.Count];
            for (int i = 0; i < config.Input.Count; i++)
            {
                if (!double.TryParse(inputStrs[i], out inputValues[i]))
                {
                    Console.WriteLine("Hatalı sayı girdiniz.");
                    return;
                }
            }

            inputValues = normalizer(inputValues);

            var output = nn.Forward(inputValues);

            Console.WriteLine("Tahmin sonucu:");
            for (int i = 0; i < output.Length; i++)
            {
                Console.WriteLine($"Output[{i}]: {output[i]:F4}");
            }
        }
        else
        {
            Console.WriteLine("Geçersiz seçim.");
        }
    }
public class TrainingData
{
    public double[] Input { get; set; }
    public double[] Output { get; set; }
}
