using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork
{
    internal class Network
    {
        readonly Layer[] layers;
        double learnRate;

        public Network(params int[] comp)
        {
            layers = new Layer[comp.Length - 1];
            for(int i = 0; i < layers.Length; i++)
            {
                layers[i] = new Layer(comp[i], comp[i + 1]);
            }
        }

        public void UpdateAllGradients(DataPoint data)
        {
            Layer output = layers[^1];
            double[] nodeValues = new double[output.outNodes];
            CalculateOutputs(data.inputs);
            for(int i = 0; i < output.outNodes; i++)
            {
                nodeValues[i] = Layer.ActivationDerivative(output.weightedInputs[i]) * CostDerivative(output.activations[i], data.expected[i]);
            }
            output.UpdateGradients(nodeValues);

            for(int layer = layers.Length - 2; layer >= 0; layer--)
            {
                Layer current = layers[layer];
                nodeValues = HiddenLayerNodeValues(current, layers[layer + 1], nodeValues);
                current.UpdateGradients(nodeValues);
            }
        }

        static double[] HiddenLayerNodeValues(Layer current, Layer oldNode, double[] oldNodeValues)
        {
            double[] newNodeValues = new double[current.outNodes];
            for (int i = 0; i < current.outNodes; i++)
            {
                newNodeValues[i] = 0;
                for(int j = 0; j < oldNode.outNodes; j++)
                {
                    newNodeValues[i] += oldNode.weights[j, i] * oldNodeValues[j];
                }
                newNodeValues[i] *= Layer.ActivationDerivative(current.activations[i]);
            }
            return newNodeValues;
        }

        public void SetLearnRate(double learnRate) => this.learnRate = learnRate;

        public double[] CalculateOutputs(double[] input)
        {
            foreach(Layer layer in layers)
            {
                input = layer.CalculateOutput(input);
            }
            return input;
        }

        static double Cost(double output, double expected)
        {
            return Math.Pow(output - expected, 2);
        }

        static double CostDerivative(double output, double expected)
        {
            return 2 * (output - expected);
        }

        public double Cost(DataPoint data)
        {
            double cost = 0;
            double[] outputs = CalculateOutputs(data.inputs);
            for(int i = 0; i < outputs.Length; i++)
            {
                cost += Cost(outputs[i], data.expected[i]);
            }
            return cost;
        }

        public double Cost(DataPoint[] data)
        {
            double cost = 0;
            foreach(DataPoint dat in data)
            {
                cost += Cost(dat);
            }
            return cost / data.Length;
        }

        void ApplyAllGradients(double learnRate)
        {
            foreach(var current in layers)
            {
                current.ApplyGradients(learnRate);
            }
        }

        void ClearAllGradients()
        {
            foreach (var current in layers) current.ClearGradients();
        }

        public void Learn(DataPoint[] data)
        {
            foreach(var i in data)
            {
                UpdateAllGradients(i);
            }
            ApplyAllGradients(learnRate / data.Length);
            ClearAllGradients();
        }
    }
}
