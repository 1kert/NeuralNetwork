using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork
{
    internal class DataPoint
    {
        public double[] inputs;
        public double[] expected;

        public static DataPoint[] MakeDataPoints(double[][] inputs, double[][] expected)
        {
            DataPoint[] arr = new DataPoint[inputs.Length];
            for(int i = 0; i < inputs.Length; i++)
            {
                arr[i] = new(inputs[i], expected[i]);
            }
            return arr;
        }

        public DataPoint(double[] inputs, double[] expected)
        {
            this.inputs = inputs;
            this.expected = expected;
        }
    }
}
