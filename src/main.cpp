#include <iostream>
#include <vector>
#include <string>
#include <algorithm>

// Function to simulate finetuning pipeline
std::vector<double> finetunePipeline(const std::vector<double>& inputs) {
    std::vector<double> outputs;
    outputs.reserve(inputs.size());

    // Simulate processing steps
    for (size_t i = 0; i < inputs.size(); ++i) {
        // Example transformation: apply a non-linear function
        double processed = inputs[i] * inputs[i] + 0.5 * inputs[i];
        outputs.push_back(processed);
    }

    return outputs;
}

// Function to generate samples before and after the fix
void generateSamples() {
    std::vector<double> originalInputs = {0.0, 0.5, 1.0, 1.5, 2.0};
    std::vector<double> originalOutputs = finetunePipeline(originalInputs);
    std::vector<double> fixedOutputs = finetunePipeline(originalInputs);

    std::cout << "=== BEFORE FIX ===" << std::endl;
    for (size_t i = 0; i < originalInputs.size(); ++i) {
        std::cout << "Input: " << originalInputs[i]
                  << " -> Output: " << originalOutputs[i] << std::endl;
    }

    std::cout << "\n=== AFTER FIX ===" << std::endl;
    for (size_t i = 0; i < originalInputs.size(); ++i) {
        std::cout << "Input: " << originalInputs[i]
                  << " -> Output: " << fixedOutputs[i] << std::endl;
    }
}

int main() {
    generateSamples();
    return 0;
}