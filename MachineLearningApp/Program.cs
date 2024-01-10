using Microsoft.ML;
using static Microsoft.ML.DataOperationsCatalog;

namespace MachineLearningApp
{
    internal class Program
    {
        static void Main(string[] args)
        {
            MLContext mlContext = new();
            TrainTestData trainTestData = CaricaDati(mlContext);
            ITransformer transformer = CreaAddestraModello(mlContext, trainTestData.TrainSet);
            Valutazione(mlContext, transformer, trainTestData.TestSet);
            ValutaRecensioneTempoReale(mlContext, transformer);

        }

        private static TrainTestData CaricaDati(MLContext mlContext)
        {
            IDataView dataView = mlContext.Data.LoadFromTextFile<Recensione>(Path.Combine(
                Environment.CurrentDirectory, "Data", "imdb_labelled.txt"), hasHeader : false);
            TrainTestData trainTest = mlContext.Data.TrainTestSplit(dataView, testFraction: 0.2);
            return trainTest;
        }

        private static ITransformer CreaAddestraModello (MLContext mlContext, IDataView dataView)
        {
            var flussoMl = mlContext.Transforms.Text.FeaturizeText(
                outputColumnName: "Opzioni", inputColumnName: nameof(Recensione.SentimentoTesto))
                .Append(mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(
                    labelColumnName: "Label", featureColumnName: "Opzioni"));
            var modello = flussoMl.Fit(dataView);
            return modello;
        }

        private static void Valutazione(MLContext mLContext, ITransformer transformer, IDataView dataView)
        {
            var predizioni = transformer.Transform(dataView);
            var metrica = mLContext.BinaryClassification.Evaluate(predizioni);

            Console.WriteLine();
            Console.WriteLine($"Accuratezza: {metrica.Accuracy:P2}");
            Console.WriteLine($"AUC (ROC): {metrica.AreaUnderRocCurve:P2}");
            Console.WriteLine($"F1Score: {metrica.F1Score:P2}");
            Console.WriteLine();
            Console.WriteLine();
            Console.WriteLine();
        }

        private static void ValutaRecensioneTempoReale(MLContext mLContext, ITransformer transformer)
        {
            Recensione recensione = new()
            {
                SentimentoTesto = "so so bad"
            };
            PredictionEngine<Recensione, PredizioneRecensione> predictionEngine = mLContext.Model.CreatePredictionEngine<Recensione, PredizioneRecensione>(transformer);
            var risultatoPredizione = predictionEngine.Predict(recensione);
            Console.WriteLine($"Esito Recensione: {recensione.SentimentoTesto} - Predizione : " +
                $"{(Convert.ToBoolean(risultatoPredizione.Predizione) ? "Positivo" : "Negativo")}" +
                $" - Probabilità: {risultatoPredizione.Probability}");
        }
    }
}
