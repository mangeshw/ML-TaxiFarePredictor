using System;
using System.IO;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Models;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;
using Microsoft.ML.Runtime.Api;
using System.Threading.Tasks;

namespace TaxiFarePrediction
{
    class Program
    {
        static readonly string _datapath = Path.Combine(Environment.CurrentDirectory, "Data", "taxi-fare-train.csv");
        static readonly string _testdatapath = Path.Combine(Environment.CurrentDirectory, "Data", "taxi-fare-test.csv");
        static readonly string _modelpath = Path.Combine(Environment.CurrentDirectory, "Data", "Model.zip");

        static async Task Main(string[] args)
        {
            try
            {
                //Train
                PredictionModel<TaxiTrip, TaxiTripFarePrediction> model = await Train();

                //Eval - Evaluate accuracy of predictions using test data
                Evaluate(model);

                //Predict
                TaxiTripFarePrediction prediction = model.Predict(TestTrips.Trip1);

                Console.WriteLine("Predicted fare: {0}, actual fare: 29.5", prediction.FareAmount);
                Console.ReadKey();
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Exception Message: {ex.Message}");
                Console.ReadKey();
            }
        }

        public static async Task<PredictionModel<TaxiTrip, TaxiTripFarePrediction>> Train()
        {
            var pipeline = new LearningPipeline
            {
                new TextLoader(_datapath).CreateFrom<TaxiTrip>(useHeader: true, separator: ','),
                new ColumnCopier(("FareAmount", "Label")),
                new CategoricalOneHotVectorizer(
                    "VendorId",
                    "RateCode",
                    "PaymentType"),
                new ColumnConcatenator(
                    "Features",
                    "VendorId",
                    "RateCode",
                    "PassengerCount",
                    "TripDistance",
                    "PaymentType"),
                new FastTreeRegressor()
            };

            PredictionModel<TaxiTrip, TaxiTripFarePrediction> model = pipeline.Train<TaxiTrip, TaxiTripFarePrediction>();
            
            await model.WriteAsync(_modelpath);

            return model;
        }

        public static void Evaluate(PredictionModel<TaxiTrip, TaxiTripFarePrediction> model)
        {
            var testData = new TextLoader(_testdatapath).CreateFrom<TaxiTrip>(useHeader: true, separator: ',');
            var evaluator = new RegressionEvaluator();
            RegressionMetrics metrics = evaluator.Evaluate(model, testData);
            Console.WriteLine($"Rms = {metrics.Rms}");
            Console.WriteLine($"RSquared = {metrics.RSquared}");
        }
    }
}
