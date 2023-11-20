using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork
{
    internal class Layer
    {
        static readonly Random rand = new();

        public int outNodes, inNodes;
        public double[,] weights;
        public double[,] gradientWeights;
        public double[] biases;
        public double[] gradientBiases;
        public double[] activations;
        public double[] weightedInputs;
        public double[] inputs;

#pragma warning disable 8618
        public Layer(int inNodes, int outNodes)
        {
            this.outNodes = outNodes;
            this.inNodes = inNodes;
            weights = new double[outNodes, inNodes];
            biases = new double[outNodes];
            gradientBiases = new double[outNodes];
            gradientWeights = new double[outNodes, inNodes];
            activations = new double[outNodes];
            weightedInputs = new double[outNodes];

            
            for(int i = 0; i < outNodes; i++)
            {
                for(int j = 0; j < inNodes; j++)
                {
                    weights[i, j] = (rand.NextDouble() * 2 - 1) / Math.Sqrt(inNodes);
                }
            }
        }

        static double Activation(double inp) => 1 / (1 + Math.Exp(-inp));
        public static double ActivationDerivative(double inp) => Math.Exp(-inp) / Math.Pow(1 + Math.Exp(-inp), 2);

        public double[] CalculateOutput(double[] input)
        {
            inputs = input;
            double[] output = new double[outNodes];
            for(int node = 0; node < outNodes; node++)
            {
                output[node] = biases[node];
                for(int inp = 0; inp < inNodes; inp++)
                {
                    output[node] += weights[node, inp] * input[inp];
                }
                weightedInputs[node] = output[node];
                output[node] = Activation(output[node]);
                activations[node] = output[node];
            }
            return output;
        }

        public void UpdateGradients(double[] nodeValues)
        {
            for(int node = 0; node < outNodes; node++)
            {
                for(int inNode = 0; inNode < inNodes; inNode++)
                {
                    gradientWeights[node, inNode] += nodeValues[node] * inputs[inNode];
                }
                gradientBiases[node] += nodeValues[node];
            }
        }

        public void ApplyGradients(double learnRate)
        {
            for (int node = 0; node < outNodes; node++)
            {
                biases[node] -= gradientBiases[node] * learnRate;

                for (int inp = 0; inp < inNodes; inp++)
                {
                    weights[node, inp] -= gradientWeights[node, inp] * learnRate;
                }
            }
        }

        public void ClearGradients()
        {
            for(int i = 0; i < outNodes; i++)
            {
                gradientBiases[i] = 0;
                for(int j = 0; j < inNodes; j++)
                {
                    gradientWeights[i, j] = 0;
                }
            }
        }
    }
}
