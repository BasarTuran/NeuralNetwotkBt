using NeuralNetworkBT;

class Program
{
    static void Main()
    {
        string configPath = "appsettings.json";

        while (true)
        {
            Console.WriteLine();
            Console.WriteLine("Yapay Sinir Ağı Uygulamasına Hoşgeldiniz.");
            Console.WriteLine("1 - Eğitim Yap");
            Console.WriteLine("2 - Tahmin Yap");
            Console.WriteLine("0 - Çıkış");
            Console.Write("Seçiminiz (0/1/2): ");

            var choice = Console.ReadLine();

            if (choice == "1")
            {
                NeuralNetworkRunner.Train(configPath);
            }
            else if (choice == "2")
            {
                NeuralNetworkRunner.Predict(configPath);
            }
            else if (choice == "0")
            {
                Console.WriteLine("Programdan çıkılıyor...");
                break;
            }
            else
            {
                Console.WriteLine("Geçersiz seçim, tekrar deneyin.");
            }
        }
    }
}