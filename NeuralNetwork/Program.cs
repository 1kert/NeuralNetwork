using NeuralNetwork;
using System;
using System.Text;

internal class Program
{
    static double[] GetInputs(string inp) => Array.ConvertAll(inp.ToCharArray(), (chr) => Convert.ToDouble(chr.ToString()));

    public static void Main()
    {
        
    }
}