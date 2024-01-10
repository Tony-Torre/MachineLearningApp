using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MachineLearningApp
{
    internal class Recensione
    {
        [LoadColumn (0)]
        public string? SentimentoTesto { get; set; }
        
        [LoadColumn (1), ColumnName("Label")]
        public bool Sentimento {  get; set; }
    }
    internal class PredizioneRecensione : Recensione
    {
        [ColumnName("PredictedLabel")]
        public bool Predizione { get; set; }
        public float Probability { get; set; }
        public float Score { get; set; }
    }
}
