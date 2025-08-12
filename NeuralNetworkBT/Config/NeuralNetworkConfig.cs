namespace NeuralNetworkBT;

public class NeuralNetworkConfig
{
   
    public InputOutputConfig Input { get; set; }
    public InputOutputConfig Output { get; set; }
    public List<HiddenLayerConfig> HiddenLayers { get; set; }
    public double LearningRate { get; set; }
    public int Epochs { get; set; }
    public bool Bias { get; set; }
    public string Normalization { get; set; }
    public string Activation { get; set; }
    public string LossFunction { get; set; }
    public string TrainingDataFile { get; set; }
    public string ModelFile { get; set; }
    public int BatchSize { get; set; } = 1;
    public int EarlyStoppingPatience { get; set; } = 10;
}

